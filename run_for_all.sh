export base_path=$1
export epochs=$2

python3 deepcor-train-o.py --base-path=$base_path --dir=64_nodes_85_c_0.01 --no-of-epochs=$epochs
python3 deepcor-train-o.py --base-path=$base_path --dir=64_nodes_85_c_0.05 --no-of-epochs=$epochs

python3 deepcor-train-o.py --base-path=$base_path --dir=64_nodes_85_c_0.001 --no-of-epochs=$epochs
python3 deepcor-train-o.py --base-path=$base_path --dir=64_nodes_85_c_0.005 --no-of-epochs=$epochs

# python3 deepcor-train-n.py --base-path=$base_path --dir=64_nodes_100_ --no-of-epochs=$epochs
# python3 deepcor-train-n.py --base-path=$base_path --dir=64_nodes_95_ --no-of-epochs=$epochs
# python3 deepcor-train-n.py --base-path=$base_path --dir=64_nodes_90_ --no-of-epochs=$epochs

# python3 deepcor-train-o.py --base-path=$base_path --dir=64_nodes_100_c --no-of-epochs=$epochs
# python3 deepcor-train-o.py --base-path=$base_path --dir=64_nodes_95_c --no-of-epochs=$epochs
# python3 deepcor-train-o.py --base-path=$base_path --dir=64_nodes_90_c --no-of-epochs=$epochs

# python3 deepcor-test.py --base-path=/$base_path --dir=64_nodes_100_c --no-of-epochs=$epochs
# python3 deepcor-test.py --base-path=/$base_path --dir=64_nodes_95_c --no-of-epochs=$epochs
# python3 deepcor-test.py --base-path=/$base_path --dir=64_nodes_90_c --no-of-epochs=$epochs

# python3 deepcor-train-o.py --base-path=$base_path --dir=64_nodes_100_d --no-of-epochs=$epochs
# python3 deepcor-train-o.py --base-path=$base_path --dir=64_nodes_95_d --no-of-epochs=$epochs
# python3 deepcor-train-o.py --base-path=$base_path --dir=64_nodes_90_d --no-of-epochs=$epochs

# python3 deepcor-test.py --base-path=/$base_path --dir=64_nodes_100_d --no-of-epochs=$epochs
# python3 deepcor-test.py --base-path=/$base_path --dir=64_nodes_95_d --no-of-epochs=$epochs
# python3 deepcor-test.py --base-path=/$base_path --dir=64_nodes_90_d --no-of-epochs=$epochs

# python3 deepcor-train-o.py --base-path=$base_path --dir=64_nodes_100_cd --no-of-epochs=$epochs
# python3 deepcor-train-o.py --base-path=$base_path --dir=64_nodes_95_cd --no-of-epochs=$epochs
# python3 deepcor-train-o.py --base-path=$base_path --dir=64_nodes_90_cd --no-of-epochs=$epochs

# python3 deepcor-test.py --base-path=/$base_path --dir=64_nodes_100_cd --no-of-epochs=$epochs
# python3 deepcor-test.py --base-path=/$base_path --dir=64_nodes_95_cd --no-of-epochs=$epochs
# python3 deepcor-test.py --base-path=/$base_path --dir=64_nodes_90_cd --no-of-epochs=$epochs



# example -: ./run_for_all.sh /export/research26/cyclone/hansika/noc_data 5
