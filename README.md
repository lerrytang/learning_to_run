NIPS Reinforcement Learning Challenge
==========================
https://www.crowdai.org/challenges/nips-2017-learning-to-run


Pre-requisite
--------

Install Anaconda from https://www.continuum.io/downloads


Environmental Setup
--------
In the root directory of this repository, input the following commands.

1. Create Conda Environment (We assume the opensim-rl environment has been activated for the rest steps.)

        conda create -n opensim-rl -c kidzik opensim git python=2.7
        source activate opensim-rl    // activate the environment, need to execute this every time

2. Install Opensim

        conda install -c conda-forge lapack git
        git clone https://github.com/stanfordnmbl/osim-rl.git
        cd osim-rl
        python setup.py install
        cd ..
        python -c "import opensim"    // confirm it runs smoothly

3. Install Tensorflow

        pip install tensorflow==1.2.0    // tensorflow-gpu is GPU is available

4. Install Keras

        git clone https://github.com/fchollet/keras.git
        cd keras
        python setup.py install
        cd ..

5. Install Other Packages (for plotting, mail notification, model saving, etc)

        pip install pandas h5py matplotlib

Examples
--------

Make directory for trials

    mkdir trials    // all trial results are saved in this folder

Training

    python dev/run.py --train
    python dev/run.py --train --visualize   // if you want to visualize the process
    python dev/run.py --train --trial_dir=trials/xxxx   // if you want to continue training

Test a trained model. It runs for 10 episodes locally.

    python dev/run.py --test --trial_dir=trials/xxxx
    python dev/run.py --test --trial_dir=trials/xxxx --visualize  // if you like to visualize the process
    python dev/run.py --test --trial_dir=trials/xxxx --submit  // to submit to the grader, need a token
