# Installation guide

A step-by-step guide from a fresh machine to a running dashboard. Short on time? The
[Quick start](../README.md#quick-start) has the same thing in a few lines.

1. [What you need](#1-what-you-need)
2. [Check the machine](#2-check-the-machine)
3. [Get the code](#3-get-the-code)
4. [Run the installer](#4-run-the-installer)
5. [First login](#5-first-login)
6. [Start your first apps](#6-start-your-first-apps)
7. [Reach it from other devices](#7-reach-it-from-other-devices)
8. [Update, stop, uninstall](#8-update-stop-uninstall)
9. [Troubleshooting](#9-troubleshooting)

Other ways to install: [Docker Compose by hand](#docker-compose-without-the-installer),
[Proxmox](proxmox.md), [TrueNAS SCALE](truenas.md), [Unraid](unraid.md) and
[WSL2](linux.md#wsl2).

## 1. What you need

| | Minimum |
| :- | :- |
| OS | Ubuntu 24.04 or 26.04, Debian 12 or 13, Fedora, RHEL/Rocky/AlmaLinux 9 or 10, Arch, or a distro based on one of them ([details](linux.md)). Ubuntu is the one tried on real hardware. |
| CPU | x86-64, 4 cores |
| RAM | 32 GB (64 GB recommended for the largest models, which keep most of their layers in system RAM) |
| Storage | 128 GB free for models and app data; an SSD is highly preferred |
| GPU | 1 or more discrete NVIDIA GPUs, GeForce GTX 1070 or newer (Pascal, compute capability 6.1+); or AMD GPUs (RX 400 series or newer) with the `amdgpu` driver, experimental |
| Access | A user with `sudo`, and internet access (Docker images and models are large downloads) |

**AMD GPUs (experimental):** on a machine with AMD GPUs and no NVIDIA ones, the installer skips the
NVIDIA driver, Container Toolkit and GPU test, and removes the NVIDIA GPU request from `compose.yaml`.
Install the *Llama Swap (Vulkan)* app from the App Center. It is untried on real AMD hardware.

You do **not** need to install Docker, the NVIDIA Container Toolkit or (on most distros) the NVIDIA
driver first: the installer checks for each one, shows what is missing and installs it after you
confirm. If you prefer to install them yourself, that works too; the installer leaves whatever is
already working alone.

What the installer sets up:

- **NVIDIA driver**: the machine's driver, which provides `nvidia-smi`.
- **Docker Engine and the Compose plugin**: runs GPU Shepherd and every app it manages.
- **NVIDIA Container Toolkit**: lets containers use the GPUs.
- **GPU Shepherd itself**: one container, started with Docker Compose, restarting after a reboot.

## 2. Check the machine

```bash
nvidia-smi          # lists your GPUs. If this fails, see "No driver yet" below
free -g             # total RAM in GB
df -h /opt /srv     # free space where the data and models will go
```

**No driver yet?** Either install it yourself in the way your distro recommends, or let the installer
do it with `--install-driver` (step 4). On RHEL-family distros, in containers (LXC) and on WSL2 it
never installs a driver; see [Linux distributions](linux.md).

**Secure Boot on?** The kernel refuses unsigned NVIDIA modules. Enrol the key when the driver
installer prompts for it, or turn Secure Boot off in the firmware. See [Secure Boot](linux.md#secure-boot).

## 3. Get the code

Install `git` if the machine doesn't have it (`sudo apt install git`, `sudo dnf install git` or
`sudo pacman -S git`), then:

```bash
git clone https://github.com/TechFreak94/GPU-Shepherd.git gpu-shepherd
cd gpu-shepherd
```

Keep this folder: it is also what updates later come from (`git pull` is done for you by
`update.sh`). Don't clone it into a temporary location. If the repository is private, Git asks for
your GitHub user name and a [personal access token](https://github.com/settings/tokens) with read
access to it; see also [Private repository](updates.md#private-repository).

## 4. Run the installer

```bash
sudo ./install.sh
```

The installer is **guided**: it shows what it found (driver, GPUs, Docker, NVIDIA Container Toolkit,
RAM), asks a few questions with a default each (press Enter to accept), then prints the plan and
changes nothing until you confirm it.

The questions:

| Question | What to answer |
| :- | :- |
| Models folder | A folder of `.gguf` model files to pick from in the dashboard (mounted read-only). With a preset, its models go to `<data folder>/models` unless you pick another folder. Leave empty if you have none and no preset. |
| Port | Where the dashboard listens. Default `8090`. |
| Settings (data) folder | Where GPU Shepherd keeps its settings, instances and app data. Default `/opt/gpu-shepherd`. Use a folder without spaces; it is mounted at the same path inside the container. |
| Preset | On the first run, a ready-made set of instances. `starter` is a Llama Swap server, ComfyUI and Open WebUI for any 8+ GB GPU. Nothing starts until you press Start. |
| Download the preset's models | About 15 GB for `starter`, in the background; safe to interrupt and resume. |
| Dashboard updates | Lets the dashboard check for updates daily and update itself on a button (a small systemd service). Recommended. |
| Docker group | Adds you to the `docker` group so `docker` works without `sudo` (takes effect at your next login). |
| Tailscale | Optional; see [step 7](#7-reach-it-from-other-devices). Off unless you ask. |

### If the driver was just installed

With `--install-driver` (or when you accept the installer's offer), a new driver needs a reboot
before it can load. The installer then stops and exits with status 3. Reboot, then run
`sudo ./install.sh` again: it picks up where it left off and does only what is still missing.

### Scripted install

Every question has an option, and `--yes` skips whatever is left, so an unattended install is one line:

```bash
sudo ./install.sh --yes --install-driver --models-dir /srv/models --port 8090 \
    --data-dir /opt/gpu-shepherd --preset starter --download-models --dashboard-updates
```

With `--yes` a preset, the model download and dashboard updates only happen when their option is
given. The full list is in [CLI and scripts](cli.md#installsh) or `./install.sh --help`.

### What it does, in order

1. Checks the NVIDIA driver (and installs it only if you ask).
2. Installs Docker Engine and the Compose plugin.
3. Installs the NVIDIA Container Toolkit and connects it to Docker.
4. Runs a test container (`docker run --gpus all ... nvidia-smi -L`) to prove containers can see the GPUs.
5. Adds you to the `docker` group.
6. Offers a preset on the first run.
7. Writes `compose.yaml` (kept if it already exists), then builds and starts GPU Shepherd.
8. Downloads the preset's models, if you chose to.
9. Sets up dashboard updates and Tailscale, if you chose them.

The installer is **safe to re-run** at any time. If it stops halfway (no network, a declined
prompt, a reboot), fix the cause and run it again.

## 5. First login

When it finishes, the installer prints the address and the first login:

```
==> GPU Shepherd is running: http://192.168.1.50:8090
    First login (also saved in /opt/gpu-shepherd/INITIAL_PASSWORD.txt):
    user: admin
    password: ...
```

Open that address in a browser on any device on the same network and log in as `admin`. Change the
password under **Settings**; the `INITIAL_PASSWORD.txt` file is deleted when you do.

Lost it? `sudo docker exec gpu-shepherd gpu-shepherd --reset-password` prints a new one at once.

GPU Shepherd controls Docker, which is root-equivalent. Keep it on a trusted network and don't
forward its port to the internet; see [step 7](#7-reach-it-from-other-devices) for a safe way in from outside.

## 6. Start your first apps

With the **starter** preset, the dashboard shows three cards: the Llama Swap, ComfyUI and Open WebUI.

1. If the models are still downloading, wait for them. Progress shows in the dashboard.
2. Press **Start** on each card. The first start of an app downloads its Docker image, so it takes
   a while; later starts are quick.
3. Turn on **Autostart** for the apps you want running after a reboot.
4. Press **Open** on a card to open the app. Starting something that needs a GPU another app is
   using shows you a plan first; confirm it and GPU Shepherd moves things around.

Without a preset, open the **App Center** and install what you want. **Install** picks a free ID and
port, mounts the right folders and connects chat apps to your model server. See
[Instance and model options](advanced.md) for every field.

## 7. Reach it from other devices

On your own network the address from step 5 is enough. To reach the dashboard and apps from
anywhere without opening ports, the installer can set up [Tailscale](https://tailscale.com):

```bash
sudo ./install.sh --tailscale                      # prints a login link
sudo TS_AUTHKEY=tskey-... ./install.sh --tailscale # or use an auth key (never saved)
sudo ./install.sh --tailscale-https                # also https://<name>.<tailnet>.ts.net
```

With dashboard updates on, you can also do all of this later in **Settings › Remote access**, which
shows a login link and this machine's Tailscale addresses. HTTPS needs certificates turned on in Tailscale's admin console. On WSL2, install Tailscale for
Windows instead. If a firewall (such as `ufw` or `firewalld`) is on, allow the dashboard port and
the ports of the apps you want to reach from your network.

## 8. Update, stop, uninstall

**Update**: use the **Update** button in the dashboard (if you enabled dashboard updates), or

```bash
sudo ./update.sh            # shows what's new, asks, then rebuilds and restarts
sudo ./update.sh --check    # only look; exit status 10 means an update is available
```

Your settings, instances, models and app data are kept, apps keep running while GPU Shepherd
restarts, and a failed update rolls back by itself. More in
[update.sh](cli.md#updatesh) and [Versions, changelog and updates](updates.md).

**Stop** (the data folder is kept), from the folder you cloned:

```bash
sudo docker compose down
```

This stops GPU Shepherd only; the apps it started keep running until you stop them in the
dashboard (stop them first if you want everything off). Start it again with `sudo docker compose up -d`.

**Uninstall**: stop the apps in the dashboard, then

```bash
sudo ./update.sh --disable-dashboard-updates   # only if you enabled dashboard updates
sudo docker compose down --rmi local           # remove GPU Shepherd and its image
sudo rm -rf /opt/gpu-shepherd          # your settings and app data: only if you want them gone
```

Models and apps' own data live in the folders you chose; delete them separately if you want the
space back. Docker, the NVIDIA driver and the Container Toolkit are left installed.

## Docker Compose without the installer

If you already have Docker, the NVIDIA driver and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html),
you can skip `install.sh`. Check the GPUs are visible to containers first:

```bash
sudo docker run --rm --gpus all ubuntu:24.04 nvidia-smi -L
```

Then either build from a clone:

```bash
git clone https://github.com/TechFreak94/GPU-Shepherd.git gpu-shepherd && cd gpu-shepherd
sudo mkdir -p /opt/gpu-shepherd
cp compose.example.yaml compose.yaml      # read the comments; adjust paths if you like
sudo docker compose up -d --build
sudo docker compose logs gpu-shepherd | grep "First start"   # the generated admin password
```

or run the published image with [compose.standalone.yaml](../compose.standalone.yaml) (see
[Docker Compose only, no clone](../README.md#docker-compose-only-no-clone)). The one rule: the data
folder must be mounted at the **same path** inside and outside the container, and so must any
model folders you want to pick from (read-only is fine). Without `install.sh` there is no
`update.sh` setup, so update with `git pull && sudo docker compose up -d --build` (or `docker compose pull`
for the image).

## 9. Troubleshooting

| Problem | What to do |
| :- | :- |
| `nvidia-smi` fails or says no devices | The driver isn't loaded. Reboot after installing it; check [Secure Boot](linux.md#secure-boot); confirm with `lspci \| grep -i nvidia` that the GPU is seen at all. |
| The installer exits with status 3 | The driver was just installed. Reboot and run `sudo ./install.sh` again. |
| "a test container could not see the GPUs" | Run `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`, then `sudo docker run --rm --gpus all ubuntu:24.04 nvidia-smi -L`. In an LXC container see [Proxmox](proxmox.md). |
| Debian: installer asks for non-free | Enable `contrib non-free non-free-firmware` in your apt sources, `sudo apt update`, run it again ([Debian](linux.md#debian)). |
| Arch: packages can't be downloaded | Your system is out of date: `sudo pacman -Syu`, reboot if the kernel changed, run it again. |
| `docker: permission denied` | You are not in the `docker` group yet: log out and back in, or use `sudo docker`. |
| The page doesn't open | `sudo docker compose ps` and `sudo docker compose logs gpu-shepherd` in the clone folder. Check the port isn't taken by another program (`sudo ss -ltnp \| grep 8090`) and that a firewall allows it. `curl http://127.0.0.1:8090/healthz` from the machine itself tells you whether it is running. |
| Forgot the password | `sudo docker exec gpu-shepherd gpu-shepherd --reset-password` |
| An app says no GPU is free | Another app holds them. Press Start and read the plan, or stop the other app. Cards show each app's GPU rule and priority (hover the ⓘ). |
| `nvidia-smi` fails inside containers after a while | After a `systemctl daemon-reload` containers can lose their GPUs. The installer adds the udev rule and device links that prevent it; re-run `sudo ./install.sh` on a machine installed earlier, then restart the apps. |
| An update failed | It rolls back by itself. By hand: `sudo ./update.sh --rollback`, or `--restore-settings`; see [When an update fails](cli.md#when-an-update-fails). |
| A model won't load or runs out of memory | Pick a smaller model or quantization, or keep part of it in system RAM; see [GPU and system RAM](advanced.md#gpu-and-system-ram). The dashboard's suggested models say whether each fits. |
| SELinux denials (Fedora, RHEL-family) | See [SELinux](linux.md#selinux-fedora-and-rhel-family). |

Still stuck? Open an issue with the installer's output, `nvidia-smi`, your distro and
`sudo docker compose logs gpu-shepherd`. Leave out passwords and tokens.
