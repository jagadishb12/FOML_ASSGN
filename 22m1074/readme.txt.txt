Team members
	Pratyay Mukherjee (22M1067)
	Avinash Tangudu (22M1061)
	Bammidi Jagadiswara Rao (22M1074)
---------------------------------------------------------------------------

IMPORTANT

---> The DIRECTORY PATH where the train, dev and test data are stored need to be given as USER INPUT at the start of the program. If proper DIRECTORY path is not provided then execution will stop.

---> It is assumed the structure of the data directory is same as provided and has "test.csv" file present (used for scaling). 

--->The above method has been tested positively in Windows 10 (64-bit) system with Python 3.7.3.


NOTE

--->For scaling Min-Max Scaling is used 
--->Please note that in Part 1 the weights were diverging despite multiple hyperparameter configuration.
	 That's why the part has been done with scaled target value (along with scaled features) to get the plots necessary.
	 Such strategies have not been employed for other parts where flexibility of network config and hyperparameter was allowed.

--->For Part 2 ADAM optimizer is used.
--->For Part 3 Momentum Gradient Descent Optimizer is used
--->Bonus Question not attempted.  