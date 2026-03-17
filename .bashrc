export OMNI_DRONES_DIR="/workspace/omni_drones"
export ISAACSIM_PATH="/workspace/isaacsim"
source "$ISAACSIM_PATH/setup_conda_env.sh"

cd $OMNI_DRONES_DIR

# check if venv exists at /opt/venv/bin/activate. If it does activate it. If it does't create a new venv at that location 
if [ -f /opt/venv/bin/activate ]; then
  echo "Activating existing venv"
  source /opt/venv/bin/activate

  # check if isaacsim exists using pip show isaacsim. If it doesn't exist echo insructions to run the install script.
  if ! pip show isaaclab > /dev/null; then
    echo "IsaacLab is not installed. You probably need to run the install script!"
    echo "Execute the bash function omni_drones_install"
  fi
else
  echo "Creating a new venv. Make sure to run the install script!" 
  echo "Execute the bash function omni_drones_install"
  /opt/conda/envs/sim/bin/python -m venv /opt/venv
  source /opt/venv/bin/activate
fi


function omni_drones_install() {
  echo "\n"
  echo "----------------------------------------"
  echo "Installing OmniDrones"
  echo "Expect some red warnings/errors, as long as the install script completes successfully, you're good to go!"
  read -rp "Press Enter to continue..."

  cd $OMNI_DRONES_DIR/docker
  ./install

  # check if pip show isaaclab is succesfful 
  if ! pip show isaaclab > /dev/null; then
    echo "----------------------------------------"
    echo "IsaacLab is not installed! There is an issue with the install script!"   
    return 1
  fi
  echo "OmniDrones installed successfully"
  echo "----------------------------------------"
  echo "Note that it's okay to see errors of the following format:"
  echo "ERROR: pip's dependency resolver does not currently take into account all the packages...."
  echo "Any other errors are not expected"
  echo "----------------------------------------"
  echo "re-source your bash using: source ~/.bashrc"
}