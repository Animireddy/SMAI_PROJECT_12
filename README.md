ABSTRACT PROPOSAL:

Project Id and title:

- Team Number: 41
- Project Id: 12 
- Team Name: Slow and Steady
- Project Title: Music Genre Classification from Lyrics

Github link: https://github.com/Animireddy/SMAI_PROJECT_12.git

Team Members:

- Animi Reddy 20161191
- Sri Keshav 20161023
- Sushman 20161143
- Raghuchandra 20161305
  
Main goal(s) of the project:

- Music Genre Classification from Lyrics

Problem definition (What is the problem? How things will be done?):

- Problem is Music Genre Classification from Lyrics

- We do it by the following way:
	SVM model:
	
	1. Read train data from input.csv file 

	2. csv file contains 2 columns - genre and song 

	3. Creating Word vectors 

	4. Convert all collected songs from input.csv to a matrix of token counts

	   - This can be done by using inbuilt CountVectorizer() function from sklearn.

	   - Matrix order: (No.of Songs in input.csv)*(Total No.of distinct Words)

	5. Give this matrix as input to Training model(like SVM) along with corresponding genre output matrix.

	6. Finally done with train model.
	

	RNN with LSTM model:
	
	Steps in LSTM modelling:
	1. Vectorize lyrics, by turning each text into either a sequence of integers or into a vector.
	2. Limit the data set to the top 50,000 words.
	3. Set the max number of words in each text at 250.
	4. Truncate and pad the input sequences so that they are all in the same length for modeling.
	5. Converting categorical labels to numbers.
	6. The first layer is the embedded layer that uses 100 length vectors to represent each word.
	7. SpatialDropout1D performs variational dropout in NLP models.
	8. The next layer is the LSTM layer with 100 memory units.
	9. The output layer must create 8 output values, one for each class.
	10. Activation function is softmax for multi-class classification.
	11. Because it is a multi-class classification problem, categorical_crossentropy is used as the loss function.



Results of the project:

- We do following steps for a given testfile 

	1. Convert all collected songs from testfile to a matrix of token counts

	   - This can be done by using inbuilt CountVectorizer() function from sklearn.

	   - Matrix order: (No.of Songs in testfile)*(Total No.of distinct Words)

	2. Call for predict function in train model and get the predicted output.

	3. With original output and predicted output we can find the accuracy obtained with the model.

Team members and tasks for each member:

- Animi Reddy: Data preprocessing for the SVM model and RNN model using NLP techniques, Code for training and predicting with the RNN model.

- Sri Keshav Kothapalli: Code for training and predicting with the SVM model, Data collection and analyzing for RNN model, slides.

- Sushman KVS: Code for training and predicting with the SVM model, Data collection and analyzing for SVM model, slides.

- Raghu Chandra: Studied research papers and Discussed with the team, slides.

Project milestones and expected timeline:

Task             													      Expected time

1. Collecting train and test data from online sources.		18-03-2019

2. Build a model for the problem 									Not yet decided(may be last week of march or first week of April)

3. Run on test data 											      	Not yet decided


