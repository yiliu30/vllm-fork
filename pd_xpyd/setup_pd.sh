# print all cmd
set -x
# alias sudo apt with sudo apt
# apt="sudo apt"

# alias apt="sudo apt"
# alias apt-get="sudo apt-get"

sudo apt install etcd -y
pip install mooncake-transfer-engine==0.3.0b3


# ### Install from Source

# ```bash
# # Install required packages
# sudo apt install git wget curl net-tools sudo iputils-ping etcd -y

# # Clone the Mooncake repository
# git clone https://github.com/kvcache-ai/Mooncake.git -b v0.3.0-beta
# cd Mooncake

# # Install dependencies
# bash dependencies.sh

# # Build and install Mooncake
# mkdir build
# cd build
# cmake ..
# make -j
# make install
# ```

### Setting up RDMA (Optional)

# To enable RDMA for high-speed data transfer, follow these steps:

# ```bash
sudo apt remove ibutils libpmix-aws
wget https://www.mellanox.com/downloads/DOCA/DOCA_v2.10.0/host/doca-host_2.10.0-093000-25.01-ubuntu2204_amd64.deb
sudo dpkg -i doca-host_2.10.0-093000-25.01-ubuntu2204_amd64.deb
sudo apt-get update
sudo apt-get -y install doca-ofed

# Check RDMA devices and network interfaces
ibdev2netdev
# Example output:
# mlx5_0 port 1 ==> ens108np0 (Up)
# mlx5_1 port 1 ==> ens9f0np0 (Up)
# ...
# ```

# RDMA requires a large amount of registered memory. It is recommended to enable Transparent Huge Pages (THP). For more details, see the [Transparent Hugepage documentation](https://docs.kernel.org/admin-guide/mm/transhuge.html).

# ```bash
# enable Transparent Huge Pages (THP)
echo always > /sys/kernel/mm/transparent_hugepage/enabled
# ```
