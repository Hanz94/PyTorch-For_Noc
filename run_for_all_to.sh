export base_path=$1
export epochs=$2

python3 deepcor-train-n.py --base-path=$base_path --dir=64_nodes_85__0.01 --no-of-epochs=$epochs
python3 deepcor-train-n.py --base-path=$base_path --dir=64_nodes_85__0.05 --no-of-epochs=$epochs

python3 deepcor-train-n.py --base-path=$base_path --dir=64_nodes_85__0.001 --no-of-epochs=$epochs
python3 deepcor-train-n.py --base-path=$base_path --dir=64_nodes_85__0.005 --no-of-epochs=$epochs



#python3 deepcor-train-n.py --base-path=$base_path --dir=64_nodes_85_ --no-of-epochs=$epochs
#python3 deepcor-train-o.py --base-path=$base_path --dir=64_nodes_85_c --no-of-epochs=$epochs

#python3 deepcor-train-n.py --base-path=$base_path --dir=64_nodes_80_ --no-of-epochs=$epochs
#python3 deepcor-train-o.py --base-path=$base_path --dir=64_nodes_80_c --no-of-epochs=$epochs


# python3 deepcor-train-o.py --base-path=$base_path --dir=64_nodes_80_cd --no-of-epochs=$epochs
# python3 deepcor-train-o.py --base-path=$base_path --dir=64_nodes_80_d --no-of-epochs=$epochs

# python3 deepcor-train-o.py --base-path=$base_path --dir=64_nodes_85_d --no-of-epochs=$epochs
# python3 deepcor-train-o.py --base-path=$base_path --dir=64_nodes_85_cd --no-of-epochs=$epochs


# python3 deepcor-test.py --base-path=/$base_path --dir=64_nodes_85_c --no-of-epochs=$epochs
# python3 deepcor-test.py --base-path=/$base_path --dir=64_nodes_80_c --no-of-epochs=$epochs


# python3 deepcor-test.py --base-path=/$base_path --dir=64_nodes_85_d --no-of-epochs=$epochs
# python3 deepcor-test.py --base-path=/$base_path --dir=64_nodes_80_d --no-of-epochs=$epochs


# python3 deepcor-test.py --base-path=/$base_path --dir=64_nodes_85_cd --no-of-epochs=$epochs
# python3 deepcor-test.py --base-path=/$base_path --dir=64_nodes_80_cd --no-of-epochs=$epochs




# example -: ./run_for_all.sh /export/research26/cyclone/hansika/noc_data 5
