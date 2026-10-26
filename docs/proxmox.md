# Proxmox VE

This hasn't been tried on real hardware yet. If something here is wrong or missing, please report it.

There are two ways to give GPU Shepherd a GPU on Proxmox VE:

| | VM with GPU passthrough | LXC container sharing the host's GPU |
|---|---|---|
| GPU is used by | the VM only | the host and several containers |
| Driver | inside the VM | on the Proxmox host, same version inside the container |
| Setup | more host steps, then a normal install | fewer host steps, but versions must match |
| Recommended | yes | if you need to share the card |

In both cases the data folder must be mounted at the same path inside and outside the GPU Shepherd
container, as usual (see [By hand](../README.md#by-hand)). That path is inside the VM or LXC
container, not on the Proxmox host.

## VM with PCIe GPU passthrough

The VM gets the whole GPU. Inside it, you install a [supported distro](linux.md) and run `install.sh`
as on any machine.

### On the Proxmox host

1. Turn on IOMMU in the firmware (VT-d on Intel, AMD-Vi / IOMMU on AMD).
2. Add the kernel option. On Intel add `intel_iommu=on`; on AMD it is on by default. Where you add it
   depends on the bootloader:
   - GRUB: edit `GRUB_CMDLINE_LINUX_DEFAULT` in `/etc/default/grub` (for example
     `quiet intel_iommu=on iommu=pt`), then run `update-grub`.
   - systemd-boot (common with ZFS on UEFI): add the option to the single line in
     `/etc/kernel/cmdline`, then run `proxmox-boot-tool refresh`.
3. Load the VFIO modules. Add these lines to `/etc/modules`:

   ```
   vfio
   vfio_iommu_type1
   vfio_pci
   ```

4. Keep the host from using the GPU. Find its IDs with `lspci -nn | grep -i nvidia` (the GPU and its
   audio device, such as `10de:xxxx` and `10de:yyyy`), then bind them to `vfio-pci` and blacklist the
   drivers in `/etc/modprobe.d/vfio.conf`:

   ```
   options vfio-pci ids=10de:xxxx,10de:yyyy
   blacklist nouveau
   blacklist nvidia
   ```

5. Run `update-initramfs -u -k all` and reboot. `lspci -nnk` should now show `Kernel driver in use:
   vfio-pci` for the GPU.

### VM settings

| Setting | Value |
|---|---|
| Machine | `q35` |
| BIOS | OVMF (UEFI), with an EFI disk |
| CPU type | `host` |
| Memory | enough for the models you want to offload to RAM; turn **Ballooning off**, since passthrough pins all of the VM's RAM |
| PCI device | the GPU, with **All Functions** and **PCI-Express** ticked |
| Primary GPU | off, unless the VM has no other display and you need it |

Then install a supported distro in the VM and run `sudo ./install.sh` there. It offers to install the
NVIDIA driver if none works. With Secure Boot on in the VM, see [Secure Boot](linux.md#secure-boot).

## LXC container sharing the host's GPU

The host and several containers can use the same GPU. The containers use the host's kernel driver, so
the driver on the host and the user-space driver inside the container must be the same version.

### On the Proxmox host

1. Install the NVIDIA driver on the host (Proxmox is Debian-based, and it needs the matching kernel
   headers). NVIDIA's `.run` installer works, or Debian's packages.
2. Check that `nvidia-smi` works and that the device files exist: `/dev/nvidia0` (one per GPU),
   `/dev/nvidiactl`, `/dev/nvidia-uvm`, `/dev/nvidia-uvm-tools` and `/dev/nvidia-caps/*`. If
   `/dev/nvidia-uvm` is missing until something uses CUDA, run `nvidia-modprobe -u -c=0` or
   load the `nvidia_uvm` module at boot.

### Container

1. Create a Debian or Ubuntu container. Under **Options > Features** turn on **Nesting**, and for an
   unprivileged container also **keyctl**, so Docker can run.
2. Pass the devices through. In the web UI use **Resources > Add > Device Passthrough** (Proxmox VE
   8.1 or later) once for each device. This writes `devN:` lines to `/etc/pve/lxc/<ID>.conf`, for
   example:

   ```
   dev0: /dev/nvidia0
   dev1: /dev/nvidiactl
   dev2: /dev/nvidia-uvm
   dev3: /dev/nvidia-uvm-tools
   dev4: /dev/nvidia-caps/nvidia-cap1
   dev5: /dev/nvidia-caps/nvidia-cap2
   ```

   For an unprivileged container you may need to add `,gid=` or `,mode=` so the container can open
   them; check the Proxmox documentation for your version.
3. Install the same driver version inside the container, without the kernel module. With NVIDIA's
   installer: `./NVIDIA-Linux-x86_64-<version>.run --no-kernel-module`. Matching distro packages work
   too.
4. In the container, clone the repository and run `sudo ./install.sh`. It detects the container and
   does not try to install a driver. It still installs Docker and the NVIDIA Container Toolkit.
5. For an unprivileged container, if the installer's GPU test fails, set `no-cgroups = true` in the
   `[nvidia-container-cli]` section of `/etc/nvidia-container-runtime/config.toml` and test again.

After a driver update on the host, update the one inside each container to the same version.
