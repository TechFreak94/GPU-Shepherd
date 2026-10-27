# Linux distributions

`install.sh` works on the distributions below. So far only Ubuntu has been used on real hardware.
The others were checked in distro containers (detection and package commands), so reports from real
machines are welcome.

The machine itself should meet the [minimum requirements](../README.md#minimum-requirements).

| Distribution | Versions | Docker | NVIDIA Container Toolkit | `--install-driver` |
|---|---|---|---|---|
| Ubuntu, and Ubuntu-based distros | 24.04, 26.04 | Ubuntu's packages (`docker.io`, `docker-compose-v2`) | NVIDIA's apt repository | `ubuntu-drivers install` |
| Debian | 12, 13 | Docker's repository (download.docker.com) | NVIDIA's apt repository | `nvidia-driver` from Debian's non-free repository |
| Fedora | current releases | Docker's repository | NVIDIA's rpm repository | RPM Fusion `akmod-nvidia` |
| RHEL, Rocky Linux, AlmaLinux | 9, 10 | Docker's repository | NVIDIA's rpm repository | not installed; see below |
| Arch Linux, and Arch-based distros | rolling | Arch's packages (`docker`, `docker-compose`, `docker-buildx`) | Arch's `nvidia-container-toolkit` package | `nvidia-open-dkms` |

Ubuntu or Debian under WSL2 also works, experimentally; see [WSL2](#wsl2).

Docker's repository means `docker-ce`, `docker-ce-cli`, `containerd.io`, `docker-buildx-plugin` and
`docker-compose-plugin`. The installer shows the plan before it changes anything.

A driver that is already installed and working is left alone. Inside a container (LXC) the installer
never installs a driver, because it has to be on the host; see [Proxmox](proxmox.md).

## Debian

The driver comes from Debian's non-free repository: `nvidia-driver`, `firmware-misc-nonfree` and the
kernel headers (`linux-headers-<arch>`, for example `linux-headers-amd64`). It needs the `contrib non-free non-free-firmware` components in your apt sources. If
they are not enabled, the installer stops and tells you how to enable them. Add them, run
`sudo apt update`, and run `install.sh` again.

## Fedora

`--install-driver` enables RPM Fusion (free and nonfree) and installs `akmod-nvidia` and
`xorg-x11-drv-nvidia-cuda`. The installer then builds the kernel module before asking you to reboot,
which can take a few minutes. Don't reboot until it says the build is done.

## RHEL, Rocky Linux and AlmaLinux

The driver is not installed automatically. The installer stops with a link to NVIDIA's driver
installation guide (the CUDA repository). Install the driver that way, reboot, and run
`sudo ./install.sh` again. If `podman-docker` is installed (it stands in for the `docker` command), the
installer replaces it with Docker CE and says so in its plan.

## Arch Linux

The installer runs `pacman -S --needed` and never does a full system upgrade. If a package can't be
downloaded (usually because your system is out of date), it tells you to run `sudo pacman -Syu`,
reboot if the kernel changed, and run `install.sh` again.

`--install-driver` installs `nvidia-open-dkms` and the headers for each installed kernel. GPUs older
than Turing (GTX 16 series and RTX 20 series) need the legacy driver from the AUR instead, which you
install by hand.

## Secure Boot

With Secure Boot on, the kernel won't load an unsigned NVIDIA module. Either enrol a key (Ubuntu,
Debian and Fedora show a MOK enrolment prompt while the driver installs, then ask for it at the next
boot) or turn Secure Boot off in the firmware. If `nvidia-smi` fails after a reboot, check this first.

## SELinux (Fedora and RHEL-family)

Docker CE has SELinux support off by default. Fedora's `moby-engine` turns it on. When Docker has
SELinux enabled, the installer:

- turns on the `container_use_devices` boolean, so containers can use the GPUs;
- writes `security_opt: [label=disable]` for the `gpu-shepherd` container in `compose.yaml`, because
  it has to reach the Docker socket.

GPU Shepherd also adds the `:z` option to the folders it mounts into the containers it creates, so
they can read and write them. It skips the system folders themselves (such as `/`, `/home`, `/opt`, `/srv`, `/mnt`) and
anything under `/run`, `/dev`, `/proc` or `/sys`, which Docker won't relabel. Subfolders such as
`/opt/gpu-shepherd` or `/srv/models` do get the label.

The dashboard's update service starts `update.sh` through `/bin/bash`, so it also works from a clone in
a home folder on SELinux hosts.

## WSL2

Experimental. The installer detects WSL2 (`/proc/sys/fs/binfmt_misc/WSLInterop`, or "microsoft" in
`/proc/version`) and prints a warning.

- **Driver:** it never installs a Linux NVIDIA driver. Install NVIDIA's current Windows driver and run
  `wsl --update` in Windows. `nvidia-smi` comes from `/usr/lib/wsl/lib`, which the installer adds to
  `PATH`.
- **Docker:** use either Docker Desktop with WSL integration turned on for the distro, or Docker Engine
  inside the distro. With Docker Desktop the installer skips the NVIDIA Container Toolkit and runtime
  setup, because Docker Desktop provides GPU support itself. Docker Engine needs systemd: set
  `systemd=true` under `[boot]` in `/etc/wsl.conf`, then run `wsl --shutdown` in Windows. With neither,
  the installer stops before changing anything and says so.
- **Updates:** the dashboard's update service needs systemd too. Without it the installer skips the
  service with a warning, and you update with `sudo ./update.sh`.
- **Ports:** open the dashboard from Windows at `http://localhost:PORT`.

## Other distributions

Install these yourself:

1. Docker with the Compose plugin
2. The NVIDIA driver
3. The [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

Then run `sudo ./install.sh --skip-deps`. It installs nothing and only writes `compose.yaml` and
starts GPU Shepherd. Or follow [By hand](../README.md#by-hand).

## Other platforms

[Proxmox](proxmox.md), [TrueNAS SCALE](truenas.md), [Unraid](unraid.md).
