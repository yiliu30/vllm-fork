# docker build -t vllm-kunshang-29 --build-arg http_proxy=http://proxy.jf.intel.com:911 --build-arg https_proxy=http://proxy.jf.intel.com:911 --build-arg HTTP_PROXY=http://proxy.jf.intel.com:911 --build-arg HTTPS_PROXY=http://proxy.jf.intel.com:911  -f docker/Dockerfile.xpu .
 


# export http_proxy=http://child-prc.intel.com:913
# export https_proxy=http://child-prc.intel.com:913
# # export  http_proxy=http://proxy.jf.intel.com:911 
# # export  https_proxy=http://proxy.jf.intel.com:911 
# docker build -f docker/Dockerfile.xpu -t vllm-xpu-env --shm-size=4g .


# # 到了docker里面用
# # export https_proxy=http://child-ir.intel.com:912
# # export http_proxy=http://child-ir.intel.com:912

docker run -it \
             --name vllm-xpu-dev \
             --network=host \
             --device /dev/dri:/dev/dri \
             -v /dev/dri/by-path:/dev/dri/by-path \
             -v /home/yiliu7/workspace:/home/yiliu7/workspace \
             --ipc=host \
             --privileged \
             xpu-dev
             
# EXPORT_PORT=6006
# echo "create docker instance ${NAME}"
# # export https_proxy=http://proxy.ims.intel.com:911
# # export http_proxy=http://proxy.ims.intel.com:911

# RUN_ARG=" -e HABANA_VISIBLE_DEVICES=all -e  https_proxy=http://proxy.ims.intel.com:911 -e http_proxy=http://proxy.ims.intel.com:911  -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add SYS_PTRACE --cap-add=sys_nice --cap-add=CAP_IPC_LOCK --ulimit memlock=-1:-1 --ipc=host --net=host --device=/dev:/dev -v /dev:/dev -v /sys/kernel/debug:/sys/kernel/debug"
# docker run -u root -it --name ${NAME} \
#     -v ${MODEL_DIR}:/hf \
#     -v /mnt/:/mnt/ \
#     --cap-add=SYSLOG \
#     ${RUN_ARG} \
#     --user root \
#     --privileged \
#     --workdir=/root \
#     --privileged \
#     -p ${EXPORT_PORT}:${EXPORT_PORT} \
#     ${IMG_NAME} 

