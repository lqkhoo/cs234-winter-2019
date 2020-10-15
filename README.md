# Stanford CS234 Reinforcement Learning (Winter 2019)

## Code
* HW1: 20.0/20.0
* HW2: 25.0/28.0
    * test4 (test_q2.Test_All) (0.0/3.0) "Test Failed: -9.207585 != -9.206065 within 0.001 delta"
* HW3: 49.0/50.0
    * test_policy_network_cheetah_sampled_action (test_all.TestAll) (0.0/0.5) "Test Failed: Lists differ: [50000, 6] != [None, 6]"
    * test_policy_network_pendulum_sampled_action (test_all.TestAll) (0.0/0.5) "Test Failed: Lists differ: [1000, 1] != [None, 1]"

## Written
* HW1: 64.0/80.0
    * 2.1 (a): -1 Minor error
    * 3.1 (a): -3 Incorrect application of ||Q*-Q~|| < epsilon
    * 3.3 (c): -1 Wrong Q*(s1, stay)
    * 3.3 (c): -1 Wrong Q*(s1, go)
    * 3.3 (c): "You forgot to calculate Q*(s1, a) here"
    * 3.4 (d): -5 Infinity bound norm not satisfied (need \tilde{Q} within epsilon from Q, in terms of the infinity norm)
    * 3.4 (d): -5 Resulting value function does not make the bound in (b) tight.
    * 3.4 (d): "The function asks you to define a ~Q. But I cannot see where they are. Second, it will be impossible to have (~Q - Q*) = 0 while making the inequality tight. The desired ~Q will assign the same value (1+gamma)/(1-gamma)2epsilon to both ~Q(s1, stay) and ~Q(s1, go)."
* HW2: 61.0/67.0
    * 5.2: -2 Missing/incorrect count for linear model
    * 6.1: -4 Incorrect
* HW3: 46.0/50.0
    * 2.1: -1 Only computes n_des / missing factor of A
    * 2.1: -3 To select \epsilon optimal arm, actually need each arm accurate to within \epsilon / 2
* Midterm: 65.5/80.0. Class stats min=9.0/med=57.0/max=77.0/mean=56.05/sdev=10.55
    * 1.2: -2 Incorrect
    * 1.10: -1 No / incorrect explanation
    * 2.1: -2 "You have shown the right thing, but did not complete the induction."
    * 3.1: -1.5 1 wrong
    * 3.2: -1 Minor error with gradient (e.g. wrong sign)
    * 4.2: -2 Proved relaxed result
    * 4.3: -4 "Applied 4b, but didn't add right terms to get recursive term."
    * 4.4: -1 Missing step

* Quiz (SCPD): 10.1/12.22
    * Q1: -1 Incorrect
    * Q4: -1 Incorrect
    * Group: "Your answer for 1 is incorrect, but I'll give credit for the explanation! Explanation for 4 is incorrect."

## Project
* Proposal: 1.0/1.0
* Final project: 100.0/100.0