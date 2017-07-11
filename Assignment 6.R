setwd("C:/Users/indra/Desktop/Spring17/ML/Assignments/Assignment 6")


library(jpeg)
library(ggplot2)
library(raster)
library(imager)
rm(list=ls())

# Download the file and save it as "Image.jpg" in the directory
img <- readJPEG("C:/Users/indra/Desktop/Spring17/ML/Assignments/Assignment 6/image2.jpg") # Read the image
imgDimension <- dim(img)


# Assign RGB channels to data frame
imgRGB <- data.frame(
  x = rep(1:imgDimension[2], each = imgDimension[1]),
  y = rep(imgDimension[1]:1, imgDimension[2]),
  R = as.vector(img[,,1]),
  G = as.vector(img[,,2]),
  B = as.vector(img[,,3])
)


# Plot the image
ggplot(data = imgRGB, aes(x = x, y = y)) + 
  geom_point(colour = rgb(imgRGB[c("R", "G", "B")]))

kClusters <- 3
kMeans <- kmeans(imgRGB[, c("R", "G", "B")], centers = kClusters)
kColours <- rgb(kMeans$centers[kMeans$cluster,])


ggplot(data = imgRGB, aes(x = x, y = y)) + 
  geom_point(colour = kColours) +
  labs(title = paste("k-Means Clustering of", kClusters, "Colours")) +
  xlab("x") +
  ylab("y") 
