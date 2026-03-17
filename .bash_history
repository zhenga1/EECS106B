omni_drones_install 
exit
omni_drones_install 
exit
pip show isaaclab
cd scripts/
python train.py algo=ppo headless=true wandb.mode=disabled
exii
exit
cd scripts/
python train.py algo=ppo headless=true wandb.mode=disabled
exit
nvidia-smi
exit
nvidia-smi
echo $CUDA_VISIBLE_DEVICES
echo $NVIDIA_VISIBLE_DEVICES
cd scripts/
python train.py algo=ppo headless=true wandb.mode=disabled
exit
cd scripts/
python train.py algo=ppo headless=true wandb.mode=disabled
nvidia-smi
nvidia-smi
exit
nvidia-smi
exit
nvidia-smi
env | grep NVIDIA
env | grep GPU
env | grep VISIBLE
exit
nvidia-smi
exit
nvidia-smi
env | grep NVIDIA
exit
nvidia-smi
unset NVIDIA_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES
nvidia-smi
env | grep CUDA
env | grep NVIDIA
env | grep GPU
exit
distrobox_enter
nvidia-smi
exit
nvidia-smi
exit
nvidia-smi
exit
nvidia-smi
exit
nvidia-smi
omni_drones_install 
source ~/.bashrc 
pip show isaaclab
cd scripts/
python train.py algo=ppo headless=true wandb.mode=disabled
exit
nvidia-smi
exit
nvidia-smi
exit
nvidia-smi
exit
nvidia-smi
exit
nvidia-smi
cd scripts/
python train.py algo=ppo headless=true wandb.mode=disabled
exit
nvidia-smi
cd scripts/
python train.py algo=ppo headless=true wandb.mode=disabled
exit
nvidia-smi
cd scripts/
python train.py algo=ppo headless=true wandb.mode=disabled
exit
nvidia-smi
exit
nvidia-smi
exit
nvidia-smi
exit
nvidia-smi
exit
nvidia-smi
exit
nvidia-smi
exit
nvidia-smi
cd scripts/
python train.py algo=ppo headless=true wandb.mode=disabled
exit
nvidia-smi
cd scripts/
python train.py algo=ppo headless=true wandb.mode=disabled
exit
nvidia-smi
exit
nvidia-smi
cd scripts/
python train.py algo=ppo headless=true wandb.mode=disabled
exit
nvidia-smi
cd scripts/
python train.py algo=ppo headless=true wandb.mode=disabled
exit
nvidia-smi
echo "PATH=$PATH"
ls -l /usr/bin/nvidia-smi || true
file /usr/bin/nvidia-smi || true
which nvidia-smi || true
ldd /usr/bin/nvidia-smi || true
ls -l /dev/nvidia* || true
nvidia-smi
nvidia-smi
python - <<PY
import torch
print("torch.cuda.is_available =", torch.cuda.is_available())
print("torch.cuda.device_count =", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device0 =", torch.cuda.get_device_name(0))
PY

exit
nvidia-smi
cd scripts/
python train.py algo=ppo headless=true wandb.mode=disabled
exit
nvidia-smi
cd scripts/
python train.py algo=ppo headless=true wandb.mode=disabled
python train.py algo=ppo headless=true wandb.mode=disabled
exit
nvidia-smi
cd scripts/
python train.py algo=ppo headless=true wandb.mode=disabled
exit
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "NVIDIA_VISIBLE_DEVICES=$NVIDIA_VISIBLE_DEVICES"
echo "VK_ICD_FILENAMES=$VK_ICD_FILENAMES"
ls -l /dev/nvidia* /dev/dri || true
nvidia-smi -L
vulkaninfo --summary
python - <<'PY'
import torch
print("torch.cuda.is_available =", torch.cuda.is_available())
print("torch.cuda.device_count =", torch.cuda.device_count())
PY

exit
nvidia-smi
vulkaninfo --summary
echo "DISPLAY=$DISPLAY"
echo "XAUTHORITY=$XAUTHORITY"
echo "XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR"
echo "VK_ICD_FILENAMES=$VK_ICD_FILENAMES"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
ls -l /usr/share/vulkan/icd.d /etc/vulkan/icd.d 2>/dev/null || true
for f in /usr/share/vulkan/icd.d/*.json /etc/vulkan/icd.d/*.json /tmp/host_vulkan/*.json; do   [ -f "$f" ] && echo "===== $f =====" && cat "$f"; done
ls -l /opt/host_nvidia_libs 2>/dev/null || true
ls -l /host_lib 2>/dev/null || true
ls -l /host_usr_lib 2>/dev/null || true
ldd /opt/host_nvidia_libs/libGLX_nvidia.so.0 2>/dev/null || true
ldd /host_usr_lib/libGLX_nvidia.so.0 2>/dev/null || true
ldd /host_usr_lib/libnvidia-glvkspirv.so.* 2>/dev/null || true
ldd /host_usr_lib/libvulkan.so.1 2>/dev/null || true
clear
ldd /host_usr_lib/libvulkan.so.1 2>/dev/null || true
echo "DISPLAY=$DISPLAY"
echo "XAUTHORITY=$XAUTHORITY"
echo "XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR"
echo "VK_ICD_FILENAMES=$VK_ICD_FILENAMES"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
ls -l /usr/share/vulkan/icd.d /etc/vulkan/icd.d 2>/dev/null || true
for f in /usr/share/vulkan/icd.d/*.json /etc/vulkan/icd.d/*.json /tmp/host_vulkan/*.json; do   [ -f "$f" ] && echo "===== $f =====" && cat "$f"; done
clear
ls -l /opt/host_nvidia_libs 2>/dev/null || true
ls -l /host_lib 2>/dev/null || true
ls -l /host_usr_lib 2>/dev/null || true
ldd /opt/host_nvidia_libs/libGLX_nvidia.so.0 2>/dev/null || true
ldd /host_usr_lib/libGLX_nvidia.so.0 2>/dev/null || true
ldd /host_usr_lib/libnvidia-glvkspirv.so.* 2>/dev/null || true
ldd /host_usr_lib/libvulkan.so.1 2>/dev/null || true
which vulkaninfo || true
vulkaninfo --summary 2>&1 | sed -n '1,120p'
exit
nvidia-smi
vulkaninfo --summary
exit
nvidia-smi
vulkaninfo --summary
echo "DISPLAY=$DISPLAY"
echo "VK_ICD_FILENAMES=$VK_ICD_FILENAMES"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
cat /tmp/host_vulkan/nvidia_icd.json
nvidia-smi
vulkaninfo --summary 2>&1 | sed -n "1,120p"
exit
nvidia-smi
vulkaninfo --summary
cat /tmp/host_vulkan/nvidia_icd.json
ldd /host_lib/libGLX_nvidia.so.0
nvidia-smi
vulkaninfo --summary 2>&1 | sed -n '1,120p'
exit
vulkaninfo 
exit
vulkaninfo 
podman exec -it omnidrones bash -lc '
cat /tmp/host_vulkan/nvidia_icd.json
echo "VK_ICD_FILENAMES=$VK_ICD_FILENAMES"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
vulkaninfo --summary 2>&1 | sed -n "1,80p"
'
cat /tmp/host_vulkan/nvidia_icd.json
echo "VK_ICD_FILENAMES=$VK_ICD_FILENAMES"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
vulkaninfo --summary 2>&1 | sed -n "1,80p"
exit
cd scripts/
python train.py algo=ppo headless=true wandb.mode=disabled
python train.py algo=ppo headless=true wandb.mode=disabled task=DroneRace
python train.py algo=ppo headless=true wandb.mode=disabled task=DroneRace
python train.py algo=ppo headless=true wandb.mode=disabled task=DroneRace
exit
