# Source this in EVERY new shell before running the DeepSDF binaries:
#
#     source ~/airway_project/DGCI/normalized_SDF_points/env.sh
#
# None of these settings survive a disconnect - they are shell state, not disk
# state. The build itself (binaries, conda env, ~/opt/pangolin) does persist.

# --- conda environment -------------------------------------------------------
export MAMBA_ROOT_PREFIX=$HOME/micromamba
eval "$($HOME/bin/micromamba shell hook -s bash)"
micromamba activate dgci

# --- shared libraries --------------------------------------------------------
# libpangolin.so lives outside the conda env; without this the binaries build
# fine but refuse to start.
export LD_LIBRARY_PATH=$HOME/opt/pangolin/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

# --- headless OpenGL ---------------------------------------------------------
# nodecpu01 has no GPU and no X server, so Pangolin uses its EGL backend with
# Mesa's llvmpipe software rasterizer.
export PANGOLIN_WINDOW_URI=headless://

# libEGL.so is only a dispatch layer; it finds the real driver through this
# manifest directory, which defaults to a system path that is empty here.
export __EGL_VENDOR_LIBRARY_DIRS=$CONDA_PREFIX/share/glvnd/egl_vendor.d
export LIBGL_DRIVERS_PATH=$CONDA_PREFIX/lib/dri
export EGL_PLATFORM=surfaceless
export GALLIUM_DRIVER=llvmpipe
export LIBGL_ALWAYS_SOFTWARE=1

# llvmpipe is reached through swrast_dri.so, not a module named llvmpipe -
# setting this override sends the loader after a file that does not exist.
unset MESA_LOADER_DRIVER_OVERRIDE

echo "dgci environment ready"
echo "  CONDA_PREFIX  $CONDA_PREFIX"
echo "  pangolin      $HOME/opt/pangolin/lib/libpangolin.so"
