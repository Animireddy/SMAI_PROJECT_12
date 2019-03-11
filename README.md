ABSTRACT PROPOSAL:

1. Read train data from input.csv file 

2. csv file contains 2 columns - genre and song 

3. Creating Word vectors 

4. Convert all collected songs from input.csv to a matrix of token counts

   - This can be done by using inbuilt CountVectorizer() function from sklearn.

   - Matrix order: (No.of Songs in input.csv)*(Total No.of distinct Words)

5. Give this matrix as input to Training model(like SVM) along with corresponding genre output matrix.

6. Convert all collected songs from testFile to a matrix of token counts

   - This can be done by using inbuilt CountVectorizer() function from sklearn.

   - Matrix order: (No.of Songs in testFile)*(Total No.of distinct Words)

7. Call for predict function in train model and get the predicted output.

8. With original output and predicted output we can find the accuracy obtained with the model.

