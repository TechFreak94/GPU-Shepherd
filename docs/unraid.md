# Unraid

This hasn't been tried on real hardware yet. If something here is wrong or missing, please report it.

Unraid isn't a normal Linux install, so `install.sh` is not used. Details change between Unraid
versions, so check the documentation for yours.

## Run it on Unraid

1. Install the **Nvidia Driver** plugin (by ich777) from Community Applications. It provides the
   driver and the NVIDIA container runtime. Pick a driver version in the plugin, then reboot.
2. Open the Unraid terminal and check that `nvidia-smi` lists your GPUs.
3. Put the source in `/mnt/user/appdata/gpu-shepherd/src`. Either download the repository zip and
   unpack it there, or, if `git` is available:

   ```bash
   git clone https://github.com/TechFreak94/GPU-Shepherd.git /mnt/user/appdata/gpu-shepherd/src
   ```

4. Choose a data folder, for example `/mnt/user/appdata/gpu-shepherd/data`. It must be mounted at the
   **same path** inside and outside the container.
5. In the source folder, copy `compose.example.yaml` to `compose.yaml` and replace `/opt/gpu-shepherd`
   with your data path (in the volume and in `SHEPHERD_DATA_DIR`).
6. Build and start it with the **Docker Compose Manager** plugin, or from the terminal:

   ```bash
   cd /mnt/user/appdata/gpu-shepherd/src
   docker compose up -d --build
   docker compose logs gpu-shepherd | grep "First start"
   ```

Then open `http://<server>:8090`; see [First login](../README.md#first-login).

## Notes

- Point the models folder at a share path that is the same everywhere, for example
  `/mnt/user/models` mounted as `/mnt/user/models` in the GPU Shepherd container. Use that path in
  the apps you install too.
- The containers GPU Shepherd creates appear in Unraid's **Docker** tab. Manage them from GPU
  Shepherd, not from there, or its plan and Unraid's view will disagree.
- Dashboard self-updates aren't available, because Unraid has no systemd. To update, refresh the
  source (`git pull`, or unpack a newer zip) and rebuild:

  ```bash
  docker compose up -d --build
  ```

- Unraid starts from a fresh root filesystem. Keep the source and data under `/mnt/user/appdata` (or
  another share) so they survive a reboot.

## Alternative: a VM

Create an Unraid VM with the GPU passed through, install a [supported distro](linux.md) in it and run
`sudo ./install.sh` there. The GPU then belongs to that VM and not to Unraid's Docker containers.
