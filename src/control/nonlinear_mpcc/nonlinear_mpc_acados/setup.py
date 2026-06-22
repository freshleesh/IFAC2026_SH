"""ament_python setup for nonlinear_mpc_acados (ported from IFAC2026_SH)."""
from glob import glob
from setuptools import find_packages, setup

package_name = 'nonlinear_mpc_acados'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.xml')
                                              + glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/config/mpc', glob('config/mpc/*.json')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hmcl',
    maintainer_email='dnrwls04@naver.com',
    description='VPMPCC + EVO-MPCC LTM acados controller (ported from IFAC2026_SH)',
    license='BSD-3-Clause',
    entry_points={
        'console_scripts': [
            'mpc_node = nonlinear_mpc_acados.mpc_node:main',
            'mpc_debug_logger = nonlinear_mpc_acados.mpc_debug_logger:main',
            'ftg_fallback_node = nonlinear_mpc_acados.ftg_fallback_node:main',
            'pp_fallback_node = nonlinear_mpc_acados.pp_fallback_node:main',
            'scan_obstacle_detector = nonlinear_mpc_acados.scan_obstacle_detector:main',
        ],
    },
)
