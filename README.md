# Final-RL-Project  
The repository contains many files, many of which were used to collect data and train a variety of models. Here is a brief description of the repository.    
    
Directories:    
charts/ - Contains charts that were generated for the final report.    
data/ - Contains some data collected that was used for the final report and visualized with charts.    
models/ - Contains saved models that have been trained for various length times steps on different target missile types  
sprites/ - Contains small pictures used for the Pygame portion of the project  
scripts/ - Variety of scripts used to generate data and plots such as the bar chart, trajectories, and episode data.  
  
Files:  
main.py - Run this with Python with Python2.10 and the modules downloaded in the requirements file. Will train an agent missile for 500,000 time steps with  
          a noisy target missile. Note the training process will take about 3 minutes to complete. A pygame window should pop up, but if it does not just look  
          for it on your monitor. Note, it works better with a Linux OS.  
  
missile_defense_env.py - The custom environment built using OpenAI's Gym framework. Modeled the rocket dynamics as a particle with restricted turn capabilities  
                         and constraints on the velocity.  
  
missile_defense_game.py - Simple game that mimmics the custom environment built. Goal is to hit the incoming missile.  
                          Controls:  
                                Right Arrow Key - Turn rocket clockwise  
                                Left Arrow Key - Turn rocket counter clockwise  
                                Up Arrow Key - Speed up missile  
                                Down Arrow Key - Slow down missile  
  
requirements.txt - All the packages needed to run the code in this repo.  
  
  