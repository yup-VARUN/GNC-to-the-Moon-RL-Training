# This is a abstraction of what a simulation episode
# would look like if wrapping and designing for a node worker on a separate vm/container

# Load/input:
# Redis_instance - To get the distributed policy and Q networks as inputs
# Message broker - To pass messages to node from master, will need ports for this



# Fork here into two processes:
    # Monitor/watchdog daemon: coordinates with the manager/master node
    # the actual episode sim
# Actual julia sim:
    # Suspend_event <multiprocessing_event>: Determined by monitor daemon; ie run if not triggered, pause if is
        # Note: this can be designed to be system wide or node independent
    # Episodes Loop:
        # This is where the sim is ran procedurally
        # while current_episode: 
            # Inside single ep:
                # Run julia_sim/julia_function(input_state, action)
                # Run julia_func1
                # Run julia_func2
                # So on...
                # End Sim(Assuming Stateless design)
                # Return state
    # return end/exitcode  

