#!/usr/bin/env python3
"""
Centerline Extractor Node for F1TENTH

Extracts track centerline from map image using skeletonization.
Based on UNICORN global_planner_node.py algorithm.

Pipeline:
    map image → binary → morphological opening → skeletonize
    → contour detection → smoothing → coordinate conversion
    → track width calculation → CSV output + visualization
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from visualization_msgs.msg import Marker, MarkerArray

import numpy as np
import cv2
import csv
import os
import math
import yaml
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from skimage.morphology import skeletonize


class CenterlineExtractor(Node):

    def __init__(self):
        super().__init__('centerline_extractor')

        # Parameters
        self.declare_parameter('map_name', '')
        self.declare_parameter('reverse', False)
        self.declare_parameter('output_csv', 'centerline.csv')
        self.declare_parameter('debug_steps', False)  # Debuging

        self.map_name = self.get_parameter('map_name').value
        self.reverse = self.get_parameter('reverse').value
        self.output_csv = self.get_parameter('output_csv').value
        self.debug_steps = self.get_parameter('debug_steps').value

        if not self.map_name:
            self.get_logger().error('map_name parameter is required!')
            return

        # canonical map storage: stack_master/maps (src side) —
        # Ubuntu: 레포 == colcon ws / Mac mini 실차: 레포가 ws/src/IFAC2026_SH 로 클론됨.
        _ws = os.path.normpath(os.path.join(next(
            p for p in os.environ.get('AMENT_PREFIX_PATH', '').split(':')
            if os.path.basename(p) == 'global_planner'), '..', '..'))
        _sm = os.path.join('src', 'system', 'stack_master')
        _repo = next(
            (r for r in (_ws, os.path.join(_ws, 'src', 'IFAC2026_SH'))
             if os.path.isdir(os.path.join(r, _sm))),
            _ws)
        self.map_dir = os.path.join(_repo, _sm, 'maps', self.map_name)
        self.get_logger().info(f'[CenterlineExtractor] map_dir: {self.map_dir}')

        # Publishers (TRANSIENT_LOCAL for latched behavior)
        latched_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.centerline_pub = self.create_publisher(
            MarkerArray, '/centerline_waypoints/markers', latched_qos)
        self.track_bounds_pub = self.create_publisher(
            MarkerArray, '/track_bounds/markers', latched_qos)

        # Run extraction
        self.get_logger().info(f'[CenterlineExtractor] Starting extraction for map: {self.map_name}')
        success = self.extract()

        if success:
            # Re-publish periodically (1Hz) for late subscribers
            self.timer = self.create_timer(1.0, self._republish)
        else:
            self.get_logger().error('[CenterlineExtractor] Extraction failed!')

    def _show_debug_plot(self, step_name: str, img: np.ndarray, cmap='gray', points=None):
        """
        각 파이프라인 단계를 시각화하는 헬퍼 함수.
        창을 닫아야 다음 파이프라인으로 넘어갑니다 (블로킹).
        """
        if not self.debug_steps:
            return

        plt.figure(figsize=(8, 8))
        plt.title(step_name)
        # ROS 맵 원점과 맞추기 위해 origin='lower' 사용
        plt.imshow(img, cmap=cmap, origin='lower')

        if points is not None:
            plt.plot(points[:, 0], points[:, 1], 'r.', markersize=3, label='Centerline Points')
            plt.legend()

        plt.axis('equal')
        plt.tight_layout()
        self.get_logger().info(f'[Debug] {step_name} 플롯을 띄웠습니다 (non-blocking).')
        plt.show(block=False)
        plt.pause(0.001)

    def extract(self) -> bool:
        """Main extraction pipeline."""
        # 1. Load map
        map_yaml_path = os.path.join(self.map_dir, f'{self.map_name}.yaml')
        if not os.path.exists(map_yaml_path):
            self.get_logger().error(f'Map YAML not found: {map_yaml_path}')
            return False

        with open(map_yaml_path, 'r') as f:
            map_data = yaml.safe_load(f)

        map_resolution = map_data['resolution']
        map_origin_x = map_data['origin'][0]
        map_origin_y = map_data['origin'][1]
        occupied_thresh = map_data.get('occupied_thresh', 0.65)
        threshold = int(occupied_thresh * 255)
        kernel = np.ones((9, 9), np.uint8)

        # centerline(스켈레톤)과 wall_boundary는 원본 [맵이름].png 기준으로 추출한다.
        # [맵이름]_modi.png는 트랙 폭(w_tr) 계산에만 사용 → optimizer가 따르는
        # 가짜 벽 제약(=경로 생성)에만 _modi가 반영되고, 중심선/경계는 실제 벽을 유지한다.
        orig_path = os.path.join(self.map_dir, map_data['image'])
        img = cv2.imread(orig_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            self.get_logger().error(f'Failed to load map image: {orig_path}')
            return False

        # Flip Y (image top-left origin → ROS bottom-left origin)
        # Same as UNICORN: cv2.flip(cv2.imread(img_path, 0), 0)
        img = cv2.flip(img, 0)
        #self._show_debug_plot('Step 1: Loaded & Flipped Original Map', img)

        # 2. Binarize (원본 = 중심선/경계용)
        bw = np.where(img > threshold, 255, 0).astype(np.uint8)
        #self._show_debug_plot('Step 2: Binarized Map', bw)

        # 3. Morphological opening (noise removal)
        opening = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)
        #self._show_debug_plot('Step 3: Morphological Opening (Noise Removal)', opening)

        # 4. Skeletonize (원본 기준)
        skeleton = skeletonize(opening, method='lee')
        skeleton_img = (skeleton * 255).astype(np.uint8)

        self.get_logger().info(f'[CenterlineExtractor] Skeleton extracted (from original image)')
        #self._show_debug_plot('Step 4: Skeletonized Map', skeleton_img)

        # 5. Extract centerline (shortest closed contour)
        try:
            centerline_cells = self._extract_centerline(skeleton_img, map_resolution)
        except Exception as e:
            self.get_logger().error(f'[CenterlineExtractor] Centerline extraction failed: {e}')
            return False

        # 6. Smooth centerline
        centerline_smooth = self._smooth_centerline(centerline_cells)
        #self._show_debug_plot('Step 5 & 6: Extracted & Smoothed Centerline', skeleton_img, points=centerline_smooth)

        # 7. Convert cells → meters
        centerline_meter = np.zeros_like(centerline_smooth, dtype=np.float64)
        centerline_meter[:, 0] = centerline_smooth[:, 0] * map_resolution + map_origin_x
        centerline_meter[:, 1] = centerline_smooth[:, 1] * map_resolution + map_origin_y

        # 8. Direction: default is CCW, reverse=true → CW
        # Determine current direction using signed area (shoelace formula)
        signed_area = self._compute_signed_area(centerline_meter)
        is_ccw = signed_area > 0

        if self.reverse:
            # Ensure CW: only flip if currently CCW
            if is_ccw:
                centerline_smooth = np.flip(centerline_smooth, axis=0)
                centerline_meter = np.flip(centerline_meter, axis=0)
            self.get_logger().info('[CenterlineExtractor] Centerline direction: CW')
        else:
            # Ensure CCW: only flip if currently CW
            if not is_ccw:
                centerline_smooth = np.flip(centerline_smooth, axis=0)
                centerline_meter = np.flip(centerline_meter, axis=0)
            self.get_logger().info('[CenterlineExtractor] Centerline direction: CCW')

        # 9. Wall boundaries (실제 벽 = 원본 이미지) — boundary CSV/마커/시각화/검증용.
        # boundary CSV(런타임 d_left/d_right, 회피 스플라인 클램핑, check_traj 검증)는
        # 반드시 실제 벽 기준이어야 한다.
        bound_right_real, bound_left_real = self._extract_wall_boundaries(
            opening, centerline_meter, map_resolution,
            map_origin_x, map_origin_y)

        # 9b. 폭(w_tr) 계산용 벽: [맵이름]_modi.png가 있으면 _modi 벽을 사용한다.
        # _modi 벽으로 폭을 좁히면 optimizer(=경로)가 그 가짜 벽을 따라가게 된다.
        # _modi가 없으면 실제(원본) 벽으로 폭을 계산한다.
        modi_path = os.path.join(self.map_dir, f'{self.map_name}_modi.png')
        use_modi = os.path.exists(modi_path)
        if use_modi:
            modi_img = cv2.imread(modi_path, cv2.IMREAD_GRAYSCALE)
            if modi_img is None:
                self.get_logger().error(
                    f'Failed to load modified map image: {modi_path}')
                return False
            modi_img = cv2.flip(modi_img, 0)
            modi_bw = np.where(modi_img > threshold, 255, 0).astype(np.uint8)
            opening_modi = cv2.morphologyEx(modi_bw, cv2.MORPH_OPEN, kernel, iterations=1)
            bound_right_w, bound_left_w = self._extract_wall_boundaries(
                opening_modi, centerline_meter, map_resolution,
                map_origin_x, map_origin_y)
            self.get_logger().info(
                '[CenterlineExtractor] Track widths from _modi image '
                '(centerline & boundaries from ORIGINAL image)')
        else:
            opening_modi = opening
            bound_right_w, bound_left_w = bound_right_real, bound_left_real

        # 10. Track width (폭 계산용 벽 기준 = _modi if exists, else 원본)
        if bound_right_w is not None and bound_left_w is not None:
            w_right, w_left = self._compute_track_widths_from_bounds(
                centerline_meter, bound_right_w, bound_left_w)
            self.get_logger().info(
                f'Asymmetric widths: w_right {w_right.min():.2f}~{w_right.max():.2f}, '
                f'w_left {w_left.min():.2f}~{w_left.max():.2f}')
        else:
            # Fallback to distance transform (symmetric, _modi 기준)
            dist_transform = cv2.distanceTransform(opening_modi, cv2.DIST_L2, 5)
            track_widths = self._compute_track_widths(
                centerline_smooth, dist_transform, map_resolution)
            w_right = track_widths
            w_left = track_widths

        # 11. Assemble complete centerline [x, y, w_right, w_left]
        centerline_complete = np.column_stack([
            centerline_meter,
            w_right,
            w_left
        ])

        n_points = len(centerline_complete)
        self.get_logger().info(f'[CenterlineExtractor] Centerline: {n_points} points')

        # 12. Write CSVs (source directory only)
        csv_path = os.path.join(self.map_dir, self.output_csv)
        self._write_csv(centerline_complete, csv_path)
        self.get_logger().info(f'[CenterlineExtractor] CSV saved: {csv_path}')

        if bound_right_real is not None:
            self._write_boundary_csv(bound_right_real, os.path.join(self.map_dir, 'boundary_right.csv'))
            self._write_boundary_csv(bound_left_real, os.path.join(self.map_dir, 'boundary_left.csv'))
            self.get_logger().info(f'[CenterlineExtractor] Boundary CSVs saved')

        # 13. Create ROS markers (실제 벽 기준)
        self._centerline_markers = self._create_centerline_markers(centerline_meter)
        if bound_right_real is not None:
            self._track_bounds_markers = self._create_track_bounds_markers(
                bound_right_real, bound_left_real)
        else:
            self._track_bounds_markers = MarkerArray()

        # Publish
        self.centerline_pub.publish(self._centerline_markers)
        self.track_bounds_pub.publish(self._track_bounds_markers)

        # 14. Matplotlib visualization (실제 벽 기준)
        self._visualize(opening, skeleton, centerline_smooth,
                        centerline_meter, bound_right_real, bound_left_real)

        return True

    def _extract_centerline(self, skeleton_img: np.ndarray,
                            map_resolution: float) -> np.ndarray:
        """
        Extract centerline from skeleton image.
        Based on UNICORN global_planner_node.py:744-815
        """
        contours, hierarchy = cv2.findContours(
            skeleton_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

        if hierarchy is None:
            raise ValueError("No contours found in skeleton")

        # Filter closed contours only
        closed_contours = []
        for i, cont in enumerate(contours):
            opened = hierarchy[0][i][2] < 0 and hierarchy[0][i][3] < 0
            if not opened:
                closed_contours.append(cont)

        if len(closed_contours) == 0:
            raise ValueError("No closed contours found in skeleton")

        self.get_logger().info(
            f'[CenterlineExtractor] Found {len(closed_contours)} closed contours')

        # Calculate perimeter of each contour, select shortest
        line_lengths = []
        for cont in closed_contours:
            length = 0.0
            for k in range(len(cont)):
                dx = cont[k][0][0] - cont[k - 1][0][0]
                dy = cont[k][0][1] - cont[k - 1][0][1]
                length += math.sqrt(dx * dx + dy * dy)
            length *= map_resolution
            line_lengths.append(length)

        min_idx = np.argmin(line_lengths)
        self.get_logger().info(
            f'[CenterlineExtractor] Selected contour {min_idx}: '
            f'{line_lengths[min_idx]:.1f}m (of {len(line_lengths)} contours)')

        smallest = np.array(closed_contours[min_idx]).flatten()
        n_points = len(smallest) // 2
        centerline = smallest.reshape(n_points, 2)

        return centerline.astype(np.float64)

    @staticmethod
    def _smooth_centerline(centerline: np.ndarray) -> np.ndarray:
        """
        Smooth centerline with Savitzky-Golay filter.
        Based on UNICORN global_planner_node.py:818-864
        """
        n = len(centerline)

        if n > 2000:
            filter_length = int(n / 200) * 10 + 1
        elif n > 1000:
            filter_length = 81
        elif n > 500:
            filter_length = 41
        else:
            filter_length = 21

        # Ensure filter_length is odd and less than n
        filter_length = min(filter_length, n - 1)
        if filter_length % 2 == 0:
            filter_length -= 1
        if filter_length < 5:
            return centerline.copy()

        # First smoothing pass
        smooth = savgol_filter(centerline, filter_length, 3, axis=0)

        # Second pass with circular shift to handle start/end boundary
        half = n // 2
        shifted = np.append(centerline[half:], centerline[:half], axis=0)
        smooth2 = savgol_filter(shifted, filter_length, 3, axis=0)

        # Merge: use second pass for start/end region of first pass
        smooth[:filter_length] = smooth2[half:half + filter_length]
        smooth[-filter_length:] = smooth2[half - filter_length:half]

        return smooth

    @staticmethod
    def _compute_signed_area(points: np.ndarray) -> float:
        """Compute signed area using shoelace formula. Positive = CCW."""
        x = points[:, 0]
        y = points[:, 1]
        return 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])

    @staticmethod
    def _compute_track_widths(centerline_cells: np.ndarray,
                              dist_transform: np.ndarray,
                              map_resolution: float) -> np.ndarray:
        """
        Get track width at each centerline point from distance transform.
        Based on UNICORN global_planner_node.py:1115-1159
        """
        rows = np.clip(centerline_cells[:, 1].astype(int),
                       0, dist_transform.shape[0] - 1)
        cols = np.clip(centerline_cells[:, 0].astype(int),
                       0, dist_transform.shape[1] - 1)
        widths = dist_transform[rows, cols] * map_resolution
        return widths
    
    @staticmethod
    def _compute_track_widths_from_bounds(centerline_meter: np.ndarray,
                                        bound_right_m: np.ndarray,
                                        bound_left_m: np.ndarray
                                        ) -> tuple:
        """
        Compute w_right and w_left by projecting boundary points onto the
        track normal direction at each centerline point.

        Previous implementation used raw Euclidean nearest-neighbour distance
        and computed bound_r/l_interp but never used them.  Both bugs are fixed
        here:
          - bound_r/l_interp (dense re-sample) is now the source for projection.
          - Width is measured along the perpendicular (normal) axis, so a corner
            wall that is ahead/behind the current position is not mistaken for
            the lateral clearance beside the current position.  This prevents
            the track from appearing narrower than it is at the entry/exit of
            straight sections, which was pushing the optimised path toward one
            wall.
        """
        # Dense-resample boundaries so the projection has fine resolution.
        bound_r = CenterlineExtractor._interpolate_2d(bound_right_m, step=0.05)
        bound_l = CenterlineExtractor._interpolate_2d(bound_left_m, step=0.05)

        n = len(centerline_meter)
        w_right = np.zeros(n)
        w_left  = np.zeros(n)

        for i in range(n):
            cl_pt  = centerline_meter[i]
            prev_i = (i - 1) % n
            next_i = (i + 1) % n

            # Unit tangent from the two neighbours
            tangent = centerline_meter[next_i] - centerline_meter[prev_i]
            t_len   = np.linalg.norm(tangent)
            tangent = tangent / t_len if t_len > 1e-6 else np.array([1.0, 0.0])

            # Rightward unit normal (rotate tangent 90° clockwise)
            normal_r = np.array([tangent[1], -tangent[0]])

            # Vectors from the current point to every boundary sample
            vecs_r = bound_r - cl_pt
            vecs_l = bound_l - cl_pt

            # Project onto normal and tangent axes
            proj_n_r =  vecs_r @ normal_r   # >0 → point is to the right
            proj_n_l = -(vecs_l @ normal_r)  # >0 → point is to the left
            proj_t_r = np.abs(vecs_r @ tangent)
            proj_t_l = np.abs(vecs_l @ tangent)

            # Only accept boundary points that are on the correct lateral side
            # AND within a 2 m tangential window (ignores walls ahead/behind).
            # min_lateral excludes near-tangential points whose proj_n ≈ 0:
            # these are boundary points that are almost directly ahead/behind
            # (not beside) the current position, causing falsely-zero width.
            window = 2.0
            min_lateral = 0.05  # [m]
            mask_r = (proj_n_r > min_lateral) & (proj_t_r < window)
            mask_l = (proj_n_l > min_lateral) & (proj_t_l < window)

            # Normal-projection distance; fall back to Euclidean if no points pass the mask.
            w_right[i] = proj_n_r[mask_r].min() if mask_r.any() else np.linalg.norm(vecs_r, axis=1).min()
            w_left[i]  = proj_n_l[mask_l].min() if mask_l.any() else np.linalg.norm(vecs_l, axis=1).min()

        # Median filter: smooths out spikes from map artifacts or corrupted
        # boundary sections (e.g., collapsed inner wall touching the centerline).
        # size=15 covers ~0.75 m of track at typical centerline density.
        from scipy.ndimage import median_filter
        w_right = median_filter(w_right.astype(float), size=15, mode='wrap')
        w_left  = median_filter(w_left.astype(float),  size=15, mode='wrap')

        # Hard floor: prevents zero-width values that make the IQP infeasible.
        # Corrupted map sections may still produce suboptimal paths, but the
        # optimizer can at least run. Remap with corrected parameters for proper widths.
        w_right = np.maximum(w_right, 0.15)
        w_left  = np.maximum(w_left,  0.15)

        return w_right, w_left

    @staticmethod
    def _interpolate_2d(points: np.ndarray, step: float) -> np.ndarray:
        """Interpolate 2D points to roughly equal step spacing."""
        diffs = np.diff(points, axis=0)
        seg_lengths = np.linalg.norm(diffs, axis=1)
        total_len = seg_lengths.sum()
        n_new = int(total_len / step)
        
        s_orig = np.concatenate([[0], np.cumsum(seg_lengths)])
        s_new = np.linspace(0, s_orig[-1], n_new)
        
        x_new = np.interp(s_new, s_orig, points[:, 0])
        y_new = np.interp(s_new, s_orig, points[:, 1])
        
        return np.column_stack([x_new, y_new])

    @staticmethod
    def _write_csv(centerline: np.ndarray, csv_path: str):
        """Write centerline to CSV. Format: x_m, y_m, w_tr_right_m, w_tr_left_m"""
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['x_m', 'y_m', 'w_tr_right_m', 'w_tr_left_m'])
            for row in centerline:
                writer.writerow([f'{row[0]:.6f}', f'{row[1]:.6f}',
                                 f'{row[2]:.4f}', f'{row[3]:.4f}'])

    def _extract_wall_boundaries(self, opening: np.ndarray,
                                    centerline_meter: np.ndarray,
                                    map_resolution: float,
                                    map_origin_x: float,
                                    map_origin_y: float):
        """
        Extract inner/outer wall contours from binary image.
        Classify as right/left relative to centerline direction.
        Returns (boundary_right_meters, boundary_left_meters) as Nx2 arrays, or (None, None).
        """
        # Find contours of the free-space (white) area
        contours, hierarchy = cv2.findContours(
            opening, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

        if hierarchy is None or len(contours) < 2:
            self.get_logger().warn('[CenterlineExtractor] Could not find 2 wall boundaries')
            return None, None

        # Select two longest contours (outer wall and inner wall)
        contour_lengths = [len(c) for c in contours]
        sorted_idx = np.argsort(contour_lengths)[::-1]

        wall_a = contours[sorted_idx[0]].reshape(-1, 2).astype(np.float64)
        wall_b = contours[sorted_idx[1]].reshape(-1, 2).astype(np.float64)

        # Convert pixel → meters
        wall_a_m = np.column_stack([
            wall_a[:, 0] * map_resolution + map_origin_x,
            wall_a[:, 1] * map_resolution + map_origin_y
        ])
        wall_b_m = np.column_stack([
            wall_b[:, 0] * map_resolution + map_origin_x,
            wall_b[:, 1] * map_resolution + map_origin_y
        ])

        # Classify right/left using cross product with centerline direction
        # Sample a few centerline points and check which side each wall is on
        n_samples = min(50, len(centerline_meter))
        step = max(1, len(centerline_meter) // n_samples)
        right_votes_a = 0

        for i in range(0, len(centerline_meter), step):
            cl_pt = centerline_meter[i]
            # Tangent at this point
            j = (i + 1) % len(centerline_meter)
            tangent = centerline_meter[j] - cl_pt
            tangent_len = np.linalg.norm(tangent)
            if tangent_len < 1e-6:
                continue
            tangent = tangent / tangent_len

            # Nearest point on wall_a
            dists_a = np.linalg.norm(wall_a_m - cl_pt, axis=1)
            nearest_a = wall_a_m[np.argmin(dists_a)]
            vec_a = nearest_a - cl_pt
            # Cross product z-component: tangent x vec > 0 means vec is to the left
            cross_a = tangent[0] * vec_a[1] - tangent[1] * vec_a[0]
            if cross_a < 0:
                right_votes_a += 1

        if right_votes_a > n_samples // (2 * step):
            boundary_right = wall_a_m
            boundary_left = wall_b_m
        else:
            boundary_right = wall_b_m
            boundary_left = wall_a_m

        self.get_logger().info(
            f'[CenterlineExtractor] Boundaries: right={len(boundary_right)} pts, '
            f'left={len(boundary_left)} pts')

        return boundary_right, boundary_left

    @staticmethod
    def _write_boundary_csv(boundary: np.ndarray, csv_path: str):
        """Write boundary contour to CSV. Format: x_m, y_m"""
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['x_m', 'y_m'])
            for row in boundary:
                writer.writerow([f'{row[0]:.6f}', f'{row[1]:.6f}'])

    @staticmethod
    def _create_centerline_markers(centerline_meter: np.ndarray) -> MarkerArray:
        """Create blue sphere markers for centerline visualization."""
        marker_array = MarkerArray()
        for i, (x, y) in enumerate(centerline_meter):
            m = Marker()
            m.header.frame_id = 'map'
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(x)
            m.pose.position.y = float(y)
            m.pose.position.z = 0.0
            m.pose.orientation.w = 1.0
            m.scale.x = 0.05
            m.scale.y = 0.05
            m.scale.z = 0.05
            m.color.r = 0.0
            m.color.g = 0.0
            m.color.b = 1.0
            m.color.a = 1.0
            marker_array.markers.append(m)
        return marker_array

    @staticmethod
    def _create_track_bounds_markers(bound_right: np.ndarray,
                                     bound_left: np.ndarray) -> MarkerArray:
        """Create track boundary markers (green=right, yellow=left)."""
        marker_array = MarkerArray()
        step = max(1, len(bound_right) // 500)  # subsample for performance

        for i in range(0, len(bound_right), step):
            m = Marker()
            m.header.frame_id = 'map'
            m.id = i
            m.ns = 'right'
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(bound_right[i, 0])
            m.pose.position.y = float(bound_right[i, 1])
            m.pose.orientation.w = 1.0
            m.scale.x = 0.03
            m.scale.y = 0.03
            m.scale.z = 0.03
            m.color.r = 0.0
            m.color.g = 1.0
            m.color.b = 0.0
            m.color.a = 0.8
            marker_array.markers.append(m)

        for i in range(0, len(bound_left), step):
            m = Marker()
            m.header.frame_id = 'map'
            m.id = i + len(bound_right)
            m.ns = 'left'
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(bound_left[i, 0])
            m.pose.position.y = float(bound_left[i, 1])
            m.pose.orientation.w = 1.0
            m.scale.x = 0.03
            m.scale.y = 0.03
            m.scale.z = 0.03
            m.color.r = 1.0
            m.color.g = 1.0
            m.color.b = 0.0
            m.color.a = 0.8
            marker_array.markers.append(m)

        return marker_array

    def _visualize(self, opening, skeleton, centerline_cells,
                   centerline_meter, bound_right, bound_left):
        """Matplotlib visualization (UNICORN debug style)."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # 1. Opening + skeleton overlay
        axes[0].imshow(opening, cmap='gray', origin='lower')
        skel_coords = np.argwhere(skeleton)
        if len(skel_coords) > 0:
            axes[0].scatter(skel_coords[:, 1], skel_coords[:, 0],
                           c='red', s=0.1, alpha=0.5)
        axes[0].set_title('Binary Map + Skeleton (red)')
        axes[0].axis('off')

        # 2. Centerline on map
        axes[1].imshow(opening, cmap='gray', origin='lower')
        axes[1].plot(centerline_cells[:, 0], centerline_cells[:, 1],
                    'b-', linewidth=1, label='Centerline')
        # Mark start with green dot
        axes[1].plot(centerline_cells[0, 0], centerline_cells[0, 1],
                    'go', markersize=8, label='Start')
        # Mark direction with arrow
        arrow_idx = min(20, len(centerline_cells) - 1)
        dx = centerline_cells[arrow_idx, 0] - centerline_cells[0, 0]
        dy = centerline_cells[arrow_idx, 1] - centerline_cells[0, 1]
        axes[1].annotate('', xy=(centerline_cells[arrow_idx, 0],
                                  centerline_cells[arrow_idx, 1]),
                         xytext=(centerline_cells[0, 0],
                                 centerline_cells[0, 1]),
                         arrowprops=dict(arrowstyle='->', color='green', lw=2))
        axes[1].legend()
        axes[1].set_title('Centerline on Map')
        axes[1].axis('off')

        # 3. Centerline + track bounds in meters
        axes[2].plot(centerline_meter[:, 0], centerline_meter[:, 1],
                    'b-', linewidth=1, label='Centerline')
        if bound_right is not None:
            step = max(1, len(bound_right) // 300)
            axes[2].plot(bound_right[::step, 0], bound_right[::step, 1],
                        'g.', markersize=1, label='Right bound')
        if bound_left is not None:
            step = max(1, len(bound_left) // 300)
            axes[2].plot(bound_left[::step, 0], bound_left[::step, 1],
                        'y.', markersize=1, label='Left bound')
        axes[2].plot(centerline_meter[0, 0], centerline_meter[0, 1],
                    'go', markersize=8, label='Start')
        axes[2].set_aspect('equal')
        axes[2].legend()
        axes[2].set_title('Centerline + Track Bounds (meters)')
        axes[2].set_xlabel('x [m]')
        axes[2].set_ylabel('y [m]')

        plt.suptitle(f'Map: {self.map_name} | Points: {len(centerline_meter)} | '
                     f'Direction: {"CW" if self.reverse else "CCW"}')
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.001)

    def _republish(self):
        """Re-publish markers periodically for late subscribers."""
        if hasattr(self, '_centerline_markers'):
            self.centerline_pub.publish(self._centerline_markers)
            self.track_bounds_pub.publish(self._track_bounds_markers)

    def cleanup_markers(self):
        """Publish DELETEALL markers to clear DDS cache before shutdown."""
        delete_msg = MarkerArray()
        m = Marker()
        m.action = Marker.DELETEALL
        delete_msg.markers.append(m)

        self.centerline_pub.publish(delete_msg)
        self.track_bounds_pub.publish(delete_msg)
        self.get_logger().info('[CenterlineExtractor] Markers cleaned up (DELETEALL)')


def main(args=None):
    rclpy.init(args=args)
    node = CenterlineExtractor()
    # Work is done in __init__ (extract + visualize). Exit immediately so that
    # the launch file's OnProcessExit handler can start trajectory_optimizer.
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
