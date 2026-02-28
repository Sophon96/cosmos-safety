# NVIDIA Cosmos Cookoff Hackathon Submission

## Motivation

Our goal is to create robotic systems that can pour liquids in a secure way. 
- Robotic labs and lab-in-the-loop AI will be the future of biology, and integrating models that reason about potential failures before they happen will ensure productivity gains don't sacrifice safety. 
- We recognize that specialized fine tuning scenarios can be limited in resources, and we aim to demonstrate how Nvidia's Cosmos models can be used to fill this gap. 

## Implementation

- We use the LeRobot So101 with an 80M parameter ACT model to be the brain of the robot.  
- We integrate Cosmos Reason to reason about trajectories on short time frames and determines whether the pouring trajectory will be successful. An unviable trajectory will be paused before the pouring commences.

## Evals

We hope to demonstrate the differences in accuracy between the Cosmos-integrated model and base ACT on 20 instances of liquid pouring. 
- One labeled cup will contain liquid to be poured into the second cup.
- The two cups will be positioned at random per iteration.
- The two policies will be evaluated on accuracy of pouring (Pass/Fail)


