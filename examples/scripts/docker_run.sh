set -x

build=${1-""}
container_name=${2-"openrlhf"}  # Provide a default container name or use the second script argument

PROJECT_PATH=$(cd $(dirname $0)/../../; pwd)
IMAGE_NAME="nvcr.io/nvidia/pytorch:23.12-py3"

if [[ "${build}" == *"b"* ]]; then
	docker image rm $IMAGE_NAME
	docker build --network=host -t $IMAGE_NAME $PROJECT_PATH/dockerfile 
else 
	docker run --runtime=nvidia --network=host -it --rm --shm-size="10g" --cap-add=SYS_ADMIN \
		-v $PROJECT_PATH:/openrlhf -v  $HOME/.cache:/root/.cache -v  $HOME/.bash_history2:/root/.bash_history \
		--name $container_name \
		$IMAGE_NAME bash
fi