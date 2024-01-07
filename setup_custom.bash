#!/bin/bash

if [[ ! -f "$(pwd)/setup_py.bash" ]]
then
  echo "please launch from the agile_flight folder!"
  exit
fi

project_path=$PWD
echo $project_path

echo "Making sure submodules are initialized and up-to-date"
git submodule update --init --recursive


echo "Compiling the agile flight environment and install the environment as python package"
cd $project_path/flightmare/flightlib/build
cmake ..
make -j10
pip install .

echo "Install RPG baseline"
cd $project_path/flightmare/flightpy/flightrl
pip install .

echo "Run the first vision demo."
cd $project_path/envtest 
python3 -m python.run_vision_demo --render 1

echo "Done!"
echo "Have a save flight!"
