export base_path=$1
export epochs=$2

python3 deepcor-train-n.py --base-path=$base_path --dir=64_nodes__FFT --no-of-epochs=$epochs
# python3 deepcor-train-n.py --base-path=$base_path --dir=64_nodes_85__0.05 --no-of-epochs=$epochs


# example -: ./run_for_all.sh /export/research26/cyclone/hansika/noc_data 5
