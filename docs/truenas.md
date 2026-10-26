# TrueNAS SCALE

This hasn't been tried on real hardware yet. If something here is wrong or missing, please report it.

TrueNAS is an appliance: it blocks package installs, so `install.sh` is not meant for it. Use a VM,
or the advanced route below. Details change between TrueNAS versions, so check the documentation for
yours.

## Recommended: a VM with GPU passthrough

1. Under **System > Advanced**, add the GPU to **Isolated GPU Device(s)**. TrueNAS can no longer use
   it, so you need another way to see the console (a second GPU, or the web UI).
2. Create a VM (**Virtualization**; called **Instances** in some versions) with enough RAM for the models you want to offload, and a disk
   with room for models.
3. In the VM's **Devices**, add the GPU (and its audio device if listed).
4. Install a [supported distro](linux.md) in the VM, clone the repository there and run
   `sudo ./install.sh`. It offers to install the NVIDIA driver.
5. Keep models and the data folder on the VM's disk, or share a dataset into the VM by NFS or SMB
   and mount it at the same path everywhere GPU Shepherd and its containers use it.

## Advanced: on the TrueNAS host (24.10 or later)

From 24.10 (Electric Eel), Apps run on Docker, so GPU Shepherd can run there too. This is not
managed by the Apps UI.

1. Go to **Apps > Configuration > Settings** and turn on **Install NVIDIA Drivers**. Wait for it to
   finish, then check that `nvidia-smi` works in the TrueNAS shell.
2. Put the source on a dataset, for example `/mnt/POOL/apps/gpu-shepherd/src`:

   ```bash
   git clone https://github.com/TechFreak94/GPU-Shepherd.git /mnt/POOL/apps/gpu-shepherd/src
   ```

3. Create a dataset for the data, for example `/mnt/POOL/apps/gpu-shepherd/data`. It must be mounted
   at the **same path** inside and outside the container. The same goes for any models folder you
   want to pick from in the UI.
4. In the source folder, copy `compose.example.yaml` to `compose.yaml`, replace `/opt/gpu-shepherd`
   with your data path (both in the volume and in `SHEPHERD_DATA_DIR`). Then, as root in the TrueNAS shell:

   ```bash
   docker compose up -d --build
   docker compose logs gpu-shepherd | grep "First start"
   ```

### Caveats

- TrueNAS doesn't manage it. It won't show in the Apps UI, and the containers GPU Shepherd creates
  won't appear as TrueNAS apps.
- TrueNAS updates can change Docker or its configuration, and may remove changes made outside the
  Apps system. Check that it still runs after an update.
- Dashboard self-updates (the systemd service) aren't supported here. To update, pull the source and
  rebuild:

  ```bash
  git pull && docker compose up -d --build
  ```
