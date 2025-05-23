#!/usr/bin/env python
import os, sys
import numpy
from os.path import join as pjoin
import shutil
import glob

try:
    from setuptools import setup, Extension, Command
    from setuptools.command.build_ext import build_ext as _build_ext
    from setuptools.command.build import build
except ImportError:
    from distutils.core import setup, Extension, Command
    from distutils.command.build_ext import build_ext as _build_ext
    from distutils.command.build import build

class CleanCommand(Command):
    """Custom distutils command to clean the .so and .pyc files."""

    user_options = [("all", "a", "")]

    def initialize_options(self):
        self.all = True
        self._clean_me = []
        self._clean_trees = []
        self._clean_exclude = []

        for root, dirs, files in list(os.walk('mavisetc')):
            for f in files:
                if f in self._clean_exclude:
                    continue
                if os.path.splitext(f)[-1] in ('.pyc', '.so', '.o',
                                               '.pyo',
                                               '.pyd', '.c', '.orig'):
                    self._clean_me.append(pjoin(root, f))
            for d in dirs:
                if d == '__pycache__':
                    self._clean_trees.append(pjoin(root, d))

        for d in ('build', 'dist', ):
            if os.path.exists(d):
                self._clean_trees.append(d)

    def finalize_options(self):
        pass

    def run(self):
        for clean_me in self._clean_me:
            try:
                os.unlink(clean_me)
            except Exception:
                pass
        for clean_tree in self._clean_trees:
            try:
                import shutil
                shutil.rmtree(clean_tree)
            except Exception:
                pass


try:
    from Cython.Build import cythonize
except ImportError:
    raise ImportError("MAVISETC requires cython to install")


class build_ext(_build_ext):
    def build_extension(self, ext):
        _build_ext.build_extension(self, ext)
        

if __name__ == "__main__":

    include_dirs = ["include",
                    numpy.get_include(),
                   ]

    cmodules = []
    cmodules += [Extension("mavisetc.utils.smoothing", 
                           ["mavisetc/utils/smoothing.pyx"], 
                           include_dirs=include_dirs)]
    ext_modules = cythonize(cmodules)

    #scripts = ['scripts/'+file for file in os.listdir('scripts/')]  
    scripts = []  

    cmdclass = {'clean': CleanCommand,
                'build_ext': build_ext}

  
    with open('mavisetc/_version.py') as f:
        exec(f.read())

    setup(
        name = "mavisetc",
        url="https://github.com/jtmendel/mavisetc",
        version= __version__,
        author="Trevor Mendel",
        author_email="trevor.mendel@anu.edu.au",
        ext_modules = ext_modules,
        cmdclass = cmdclass,
        scripts = scripts, 
        packages=["mavisetc",
                  "mavisetc.instruments",
                  "mavisetc.telescopes",
                  "mavisetc.detectors",
                  "mavisetc.background",
                  "mavisetc.sources",
                  "mavisetc.utils",
                  "mavisetc.filters"],
        license="LICENSE",
        description="MAVIS Exposure Time Calculator",
        package_data={"": ["README.md", "LICENSE"],
                      "mavisetc": ["mavisetc/data/mavis/PSF_PC10_2024-06-27.fits",
                              "mavisetc/data/mavis/PSF_PC25_2024-06-27.fits",
                              "mavisetc/data/mavis/PSF_PC50_2024-06-27.fits",
                              "mavisetc/data/mavis/PSF_PC50_2025-05-14.fits",
                              "mavisetc/data/mavis/PSF_PC75_2024-06-27.fits",
                              "mavisetc/data/mavis/PSF_PC90_2024-06-27.fits",
                              "mavisetc/data/mavis/PSF_PC50_spec-25mas_2025-04-03.fits",
                              "mavisetc/data/mavis/PSF_PC50_spec-50mas_2025-04-03.fits",
                              "mavisetc/data/mavis/PSF_PC50_spec-25mas_2025-05-14.fits",
                              "mavisetc/data/mavis/PSF_PC50_spec-50mas_2025-05-14.fits",
                              "mavisetc/data/mavis/PSF_spec-25mas_2025-05-14.fits",
                              "mavisetc/data/mavis/PSF_spec-50mas_2025-05-14.fits",
                              "mavisetc/data/mavis/PSF_img_2025-05-14.fits",
                              "mavisetc/data/mavis/PSF_TLRatmo_2024-08-28.fits",
                              "mavisetc/data/mavis/MAVIS_throughput_spec_2024-05-14.csv",
                              "mavisetc/data/mavis/MAVIS_throughput_spec_2025-03-06.csv",
                              "mavisetc/data/mavis/mavis_AOM_throughput.csv",
                              "mavisetc/data/mavis/mavis_AOM_throughput_2024-09-27_img.csv",
                              "mavisetc/data/mavis/mavis_AOM_throughput_2024-09-27_spec.csv",
                              "mavisetc/data/mavis/mavis_AOM_throughput_2025-03-11_img.csv",
                              "mavisetc/data/mavis/mavis_AOM_throughput_2025-03-11_spec.csv",
                              "mavisetc/data/mavis/mavis_AOM_throughput_2025-03-14_img.csv",
                              "mavisetc/data/mavis/mavis_AOM_throughput_2025-03-14_spec.csv",
                              "mavisetc/data/mavis/notch_throughput.csv",
                              "mavisetc/data/ref_sky_dark.fits",
                              "mavisetc/data/ref_sky_grey.fits",
                              "mavisetc/data/ref_sky_bright.fits",
                              "mavisetc/data/E2V290_QE.csv",
                              "mavisetc/data/E2V290_QE_min.csv",
                              "mavisetc/data/E2V250_QE.csv",
                              "mavisetc/data/STA1600_QE.csv",
                              "mavisetc/data/COSMOS_QE.csv",
			                  "mavisetc/data/UT4_M1_reflect.csv",
			                  "mavisetc/data/UT4_M2_reflect.csv",
			                  "mavisetc/data/UT4_M3_reflect.csv",
                              "mavisetc/data/kc_templates/elliptical_template.fits",
                              "mavisetc/data/kc_templates/s0_template.fits",
                              "mavisetc/data/kc_templates/sa_template.fits",
                              "mavisetc/data/kc_templates/sb_template.fits",
                              "mavisetc/data/kc_templates/sc_template.fits",
                              "mavisetc/data/kc_templates/starb1_template.fits",
                              "mavisetc/data/kc_templates/starb2_template.fits",
                              "mavisetc/data/kc_templates/starb3_template.fits",
                              "mavisetc/data/kc_templates/starb4_template.fits",
                              "mavisetc/data/kc_templates/starb5_template.fits",
                              "mavisetc/data/kc_templates/starb6_template.fits",
							  "mavisetc/data/lasd_templates/GALEX0330m2816.ascii",
							  "mavisetc/data/lasd_templates/GALEX0331m2814.ascii",
							  "mavisetc/data/lasd_templates/GALEX0332m2801.ascii",
							  "mavisetc/data/lasd_templates/GALEX0332m2811.ascii",
							  "mavisetc/data/lasd_templates/GALEX0333m2821.ascii",
							  "mavisetc/data/lasd_templates/GALEX1000p0157.ascii",
							  "mavisetc/data/lasd_templates/GALEX1001p0233.ascii",
							  "mavisetc/data/lasd_templates/GALEX1417p5228.ascii",
							  "mavisetc/data/lasd_templates/GALEX1417p5305.ascii",
							  "mavisetc/data/lasd_templates/GALEX1418p5217.ascii",
							  "mavisetc/data/lasd_templates/GALEX1418p5218.ascii",
							  "mavisetc/data/lasd_templates/GALEX1418p5307.ascii",
							  "mavisetc/data/lasd_templates/GALEX1419p5315.ascii",
							  "mavisetc/data/lasd_templates/GALEX1420p5243.ascii",
							  "mavisetc/data/lasd_templates/GALEX1423p5246.ascii",
							  "mavisetc/data/lasd_templates/GALEX1434p3532.ascii",
							  "mavisetc/data/lasd_templates/GALEX1436p3456.ascii",
							  "mavisetc/data/lasd_templates/GALEX1437p3445.ascii",
							  "mavisetc/data/lasd_templates/GALEX1717p5944.ascii",
							  "mavisetc/data/lasd_templates/GP0303m0759.ascii",
							  "mavisetc/data/lasd_templates/GP0749p3337.ascii",
							  "mavisetc/data/lasd_templates/GP0751p1638.ascii",
							  "mavisetc/data/lasd_templates/GP0822p2241.ascii",
							  "mavisetc/data/lasd_templates/GP0911p1831.ascii",
							  "mavisetc/data/lasd_templates/GP0917p3152.ascii",
							  "mavisetc/data/lasd_templates/GP0927p1740.ascii",
							  "mavisetc/data/lasd_templates/GP1009p2916.ascii",
							  "mavisetc/data/lasd_templates/GP1018p4106.ascii",
							  "mavisetc/data/lasd_templates/GP1032p2717.ascii",
							  "mavisetc/data/lasd_templates/GP1054p5238.ascii",
							  "mavisetc/data/lasd_templates/GP1122p6154.ascii",
							  "mavisetc/data/lasd_templates/GP1133p6514.ascii",
							  "mavisetc/data/lasd_templates/GP1137p3524.ascii",
							  "mavisetc/data/lasd_templates/GP1205p2620.ascii",
							  "mavisetc/data/lasd_templates/GP1219p1526.ascii",
							  "mavisetc/data/lasd_templates/GP1244p0216.ascii",
							  "mavisetc/data/lasd_templates/GP1249p1234.ascii",
							  "mavisetc/data/lasd_templates/GP1339p1516.ascii",
							  "mavisetc/data/lasd_templates/GP1424p4217.ascii",
							  "mavisetc/data/lasd_templates/GP1440p4619.ascii",
							  "mavisetc/data/lasd_templates/GP1454p4528.ascii",
							  "mavisetc/data/lasd_templates/GP1514p3852.ascii",
							  "mavisetc/data/lasd_templates/GP1543p3446.ascii",
							  "mavisetc/data/lasd_templates/GP1559p0841.ascii",
							  "mavisetc/data/lasd_templates/GP2237p1336.ascii",
							  "mavisetc/data/lasd_templates/HARO11.ascii",
							  "mavisetc/data/lasd_templates/J0007p0226.ascii",
							  "mavisetc/data/lasd_templates/J0021p0052.ascii",
							  "mavisetc/data/lasd_templates/J0055m0021.ascii",
							  "mavisetc/data/lasd_templates/J010534p234960.ascii",
							  "mavisetc/data/lasd_templates/J0150p1308.ascii",
							  "mavisetc/data/lasd_templates/J015208m043117.ascii",
							  "mavisetc/data/lasd_templates/J0159p0751.ascii",
							  "mavisetc/data/lasd_templates/J020819m040136.ascii",
							  "mavisetc/data/lasd_templates/J0213p1259.ascii",
							  "mavisetc/data/lasd_templates/J0232m0426.ascii",
							  "mavisetc/data/lasd_templates/J0808p3948.ascii",
							  "mavisetc/data/lasd_templates/J0815p2156.ascii",
							  "mavisetc/data/lasd_templates/J0820p5431.ascii",
							  "mavisetc/data/lasd_templates/J0901p2119.ascii",
							  "mavisetc/data/lasd_templates/J0919p4906.ascii",
							  "mavisetc/data/lasd_templates/J0921p4509.ascii",
							  "mavisetc/data/lasd_templates/J0925p1403.ascii",
							  "mavisetc/data/lasd_templates/J0926p4427.ascii",
							  "mavisetc/data/lasd_templates/J0938p5428.ascii",
							  "mavisetc/data/lasd_templates/J1011p1947.ascii",
							  "mavisetc/data/lasd_templates/J1025p3622.ascii",
							  "mavisetc/data/lasd_templates/J1032p4919.ascii",
							  "mavisetc/data/lasd_templates/J1046p5827.ascii",
							  "mavisetc/data/lasd_templates/J110359p483456.ascii",
							  "mavisetc/data/lasd_templates/J110506p594741.ascii",
							  "mavisetc/data/lasd_templates/J1112p5503.ascii",
							  "mavisetc/data/lasd_templates/J1113p2930.ascii",
							  "mavisetc/data/lasd_templates/J1121p3806.ascii",
							  "mavisetc/data/lasd_templates/J1127p4610.ascii",
							  "mavisetc/data/lasd_templates/J1144p4012.ascii",
							  "mavisetc/data/lasd_templates/J1152p3400.ascii",
							  "mavisetc/data/lasd_templates/J1154p2443.ascii",
							  "mavisetc/data/lasd_templates/J1205p4551.ascii",
							  "mavisetc/data/lasd_templates/J121948p481411.ascii",
							  "mavisetc/data/lasd_templates/J1233p4959.ascii",
							  "mavisetc/data/lasd_templates/J1242p4851.ascii",
							  "mavisetc/data/lasd_templates/J1243p4646.ascii",
							  "mavisetc/data/lasd_templates/J124619p444902.ascii",
							  "mavisetc/data/lasd_templates/J1248p4259.ascii",
							  "mavisetc/data/lasd_templates/J1256p4509.ascii",
							  "mavisetc/data/lasd_templates/J1333p6246.ascii",
							  "mavisetc/data/lasd_templates/J1349p5631.ascii",
							  "mavisetc/data/lasd_templates/J1355p1457.ascii",
							  "mavisetc/data/lasd_templates/J1355p4651.ascii",
							  "mavisetc/data/lasd_templates/J1414p0540.ascii",
							  "mavisetc/data/lasd_templates/J1416p1223.ascii",
							  "mavisetc/data/lasd_templates/J142535p524902.ascii",
							  "mavisetc/data/lasd_templates/J1428p1653.ascii",
							  "mavisetc/data/lasd_templates/J1429p0643.ascii",
							  "mavisetc/data/lasd_templates/J1442m0209.ascii",
							  "mavisetc/data/lasd_templates/J1455p6107.ascii",
							  "mavisetc/data/lasd_templates/J1457p2232.ascii",
							  "mavisetc/data/lasd_templates/J1503p3644.ascii",
							  "mavisetc/data/lasd_templates/J1521p0759.ascii",
							  "mavisetc/data/lasd_templates/J1525p0757.ascii",
							  "mavisetc/data/lasd_templates/J1612p0817.ascii",
							  "mavisetc/data/lasd_templates/J2103m0728.ascii",
							  "mavisetc/data/lasd_templates/KISSR1084.ascii",
							  "mavisetc/data/lasd_templates/KISSR1567.ascii",
							  "mavisetc/data/lasd_templates/KISSR1578.ascii",
							  "mavisetc/data/lasd_templates/KISSR242.ascii",
							  "mavisetc/data/lasd_templates/Tol.ascii",
							  "mavisetc/data/lasd_templates/lasd_measurements.cat",
                              "mavisetc/data/lamp_templates/Cd_to_Focus.csv",
                              "mavisetc/data/lamp_templates/Zn_to_Focus.csv",
                              "mavisetc/data/lamp_templates/Hector_Ne_to_Focus.csv",
                              "mavisetc/data/lamp_templates/Hector_Xe_to_Focus.csv",
                              "mavisetc/data/lamp_templates/LDLS_100um_Core_to_Focus_Etalon.csv",
                              "mavisetc/data/lamp_templates/LDLS_100um_Core_to_Focus.csv",
                              "mavisetc/data/lamp_templates/Thorlabs_SLS201L_QTH_to_Focus.csv",
                              "mavisetc/data/lamp_templates/Thorlabs_OSL2IR_QTH_to_Focus.csv",
                              "mavisetc/data/lamp_templates/LDLS_100um_Core_Fibre_to_Focus_With_Spectrograph_Grid_Pinholes.csv",
                              "mavisetc/data/lamp_templates/Thorlabs_SLS201L_QTH_Fibre_to_Focus_With_Spectrograph_Grid_Pinholes.csv",
                              "mavisetc/data/lamp_templates/Thorlabs_OSL2IR_QTH_Fibre_to_Focus_With_Spectrograph_Grid_Pinholes.csv",
                              "mavisetc/data/lamp_templates/ACM_Pinhole_MGG_Lamps.csv",
                              "mavisetc/data/stellar_templates/MAVIS_stellar_library.fits",
                              "mavisetc/data/filters/cousins_i.dat",
                              "mavisetc/data/filters/cousins_r.dat",
                              "mavisetc/data/filters/johnson_b.dat",
                              "mavisetc/data/filters/johnson_u.dat",
                              "mavisetc/data/filters/johnson_v.dat",
                              "mavisetc/data/filters/sdss_g.dat",
                              "mavisetc/data/filters/sdss_i.dat",
                              "mavisetc/data/filters/sdss_r.dat",
                              "mavisetc/data/filters/sdss_u.dat",
                              "mavisetc/data/filters/sdss_z.dat",
                              "mavisetc/data/filters/mavis_u.dat",
                              "mavisetc/data/filters/mavis_g.dat",
                              "mavisetc/data/filters/mavis_r.dat",
                              "mavisetc/data/filters/lsst_g.dat",
                              "mavisetc/data/filters/lsst_i.dat",
                              "mavisetc/data/filters/lsst_r.dat",
                              "mavisetc/data/filters/lsst_u.dat",
                              "mavisetc/data/filters/lsst_z.dat",
                              "mavisetc/data/filters/prime_g.dat",
                              "mavisetc/data/filters/prime_i.dat",
                              "mavisetc/data/filters/prime_r.dat",
                              "mavisetc/data/filters/prime_u.dat",
                              "mavisetc/data/filters/prime_z.dat",
                              "mavisetc/data/filters/asahi_g.dat",
                              "mavisetc/data/filters/asahi_i.dat",
                              "mavisetc/data/filters/asahi_r.dat",
                              "mavisetc/data/filters/asahi_u.dat",
                              "mavisetc/data/filters/asahi_z.dat",
                              "mavisetc/data/filters/stromgren_b.dat",
                              "mavisetc/data/filters/stromgren_u.dat",
                              "mavisetc/data/filters/stromgren_v.dat",
                              "mavisetc/data/filters/stromgren_y.dat",
                              "mavisetc/data/filters/suprimecam_ia427.dat",
                              "mavisetc/data/filters/suprimecam_ia445.dat",
                              "mavisetc/data/filters/suprimecam_ia464.dat",
                              "mavisetc/data/filters/suprimecam_ia484.dat",
                              "mavisetc/data/filters/suprimecam_ia505.dat",
                              "mavisetc/data/filters/suprimecam_ia527.dat",
                              "mavisetc/data/filters/suprimecam_ia550.dat",
                              "mavisetc/data/filters/suprimecam_ia574.dat",
                              "mavisetc/data/filters/suprimecam_ia598.dat",
                              "mavisetc/data/filters/suprimecam_ia624.dat",
                              "mavisetc/data/filters/suprimecam_ia651.dat",
                              "mavisetc/data/filters/suprimecam_ia679.dat",
                              "mavisetc/data/filters/suprimecam_ia709.dat",
                              "mavisetc/data/filters/suprimecam_ia738.dat",
                              "mavisetc/data/filters/suprimecam_ia767.dat",
                              "mavisetc/data/filters/suprimecam_ia797.dat",
                              "mavisetc/data/filters/suprimecam_ia827.dat",
                              "mavisetc/data/filters/suprimecam_ia856.dat",
                              "mavisetc/data/filters/suprimecam_ia907.dat",
                              "mavisetc/data/filters/hst_f550m.dat",
                              "mavisetc/data/filters/hst_f547m.dat",
                              "mavisetc/data/filters/jwst_f070w.dat",
                              "mavisetc/data/filters/jwst_f090w.dat",
                              "mavisetc/data/filters/hst_acs_f435w.dat",
                              "mavisetc/data/filters/hst_acs_f475w.dat",
                              ]},
        include_package_data=True,
        zip_safe=False,
    )



