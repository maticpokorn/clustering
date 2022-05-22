# clustering
A school project, that creates a defined number of clusters out of similar jpeg images, similarities of which are computed using a pre-trained PyTorch neural network.
The clustering uses the K-medoids algorithm.

# running the code
The code is run either by typing in the terminal ``python naloga1.py`` without any arguments, which will run the code with 5 clusters and on a folder of 50 images called ``stock_images``,
or by also specifying the number of clusters and the image folder that you want to use, for example: ``python naloga1.py 4 my_images``

# output
The output of the program are K png files, where K is the number of clusters specified. Each png shows all the images corresponding to the same cluster.
