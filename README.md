# Java Machine Learning Toolkit

A collection of libraries for data ingress & normalization and machine learning.

Machine learning techniques available include:

- Decision trees (classification and regression)
- Linear Regression
- Logistic Regression
- Naive Bayes Document Classification
- Perceptron and Kernelized Perceptron
- SVM (using LIBSVM)
- K-Means clustering and Gaussian Mixture Models

## Requirements

This project was coded using Java 8 and the Eclipse IDE. To build
and run the project you must have the following installed.

- Java 8 SDK (author used Java 8 SDK Update 151, 64-bit). You can download
a JDK installer from http://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html.
- Eclipse IDE for Java Developers, which can be downloaded from http://www.eclipse.org/downloads/packages/eclipse-ide-java-developers/oxygen2. The author used Eclipse Oxygen.2 Release (4.7.2).

You can find detailed installation instructions for Eclipse in their online
documentation: https://wiki.eclipse.org/Eclipse/Installation

## Setting up the project

1. Checkout the project files to a convenient location on your computer.
2. Open Eclipse and add the project to your workspace.
	- `File` -> `Import` -> `General` -> `Existing projects into workspace`
	- Browse to the location of the folder where you unzipped the project files.
	- Ensure that the `MachineLearning` project is checked and hit `Finish`

### Setup Troubleshooting

If you are unable to import the project as described above, please try the
following steps.

1. Open Eclipse and create a new Java project: `File` -> `New` -> `Java Project`
2. Import the project files
	- Click `File` -> `Import` -> `General` -> `File System`
	- For `From Directory`, select the folder containing the project files.
	- For `Into folder`, choose the project that you just created
	- Click Finished

## Building the project

### Building

By default, your Eclipse project should be set to `Build Automatically`. If not,
you can manually build the project by pressing `Ctrl + B` or navigating to the
`Project` dropdown and selecting `Build All`.

#### Library Dependencies

The additional libraries that this project depends on are:

- [Commons Math](http://commons.apache.org/proper/commons-math/)
- [GRAL](https://github.com/eseifert/gral)
- [XChart](https://github.com/timmolter/xchart)
- [LibSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/)

The jar files for these libraries can be found inside the `lib` folder. The build path
should already be configured to include them, but if it is not you can right-click
on the project and choose `Properties` -> `Java Build Path` -> `Add JARs`, and select
all the `.jar` files in the `lib` folder.
