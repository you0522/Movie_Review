# Movie_Review_Test

This project predicts whether a given tweet is about a real disaster or not using Natural Langauge Procecssing (NLP).
This project predicts the score of a movie based on the reviews from Rotten Tomatoes Dataset using Natural Language Processing (NLP).

## Getting Started

Use any python IDE to open the project. I personally use Jupyter Notebook from Anaconda, but a good alternative is Google Colab. You can download both Anaconda and Jupyter Notebook from the following links:
* [Anaconda](https://www.anaconda.com/distribution/) - The Data Science Platform for Python/R
* [Jupyter Notebook](https://jupyter.org/) - An Open-source Web Application
For more about Google Colab, go to:
* [Google Colaboratory](https://colab.research.google.com/notebooks/welcome.ipynb) - Free Jupyter notebook environment that runs entirely in the cloud.

### Data

The Rotten Tomatoes dataset for this project is available on Kaggle which is a huge community space for data scientists. Click the following link to download the dataset:
* [Movie_Review_Dataset](https://www.kaggle.com/c/movie-review-sentiment-analysis-kernels-only/data) - Kaggle Movie Review Dataset

### Installation

Before running the program, type the following command to install the libraries that the project depends on

```
pip install tensorflow, matplotlib, scikit, numpy
```
Or simply type the following:

```
pip install -r requirements.txt
```

## Running the tests

- The description of each function is located on top of them. Please read them before running to understand the overall structure of the project. <br/>
- This project uses different models to classify a review (in other words, sentnece) to 5 different categories, which are:<br/>

  * 0 - Negative
  * 1 - Somewhat Negative
  * 2 - Neutral
  * 3 - Somewhat Positive
  * 4 - Positive

- The following shows the prediction from all models:

... (The image is to be posted later...)
... (Description of output is to be posted later...)

- For more detail, please read the descriptions on top of each function, and go to **main.ipynb**. The Neural_Network class is designed to let the users customize their own model. Let me know if you can come up with a model that givse 100% accuracy!.<br/>
- I also added a **py** file for the main functionin in the **src** directory if you want to run it using IDE

## Deployment

Download other dataset from online (Ex: Kaggle) and insert the data to the model in order to test its accuracy.<br/>
* [Kaggle](https://www.kaggle.com/) - The Machine Learning and Data Science Community 

## Built With

* [Python3.7](https://www.python.org/) - The Programming Language

## Author

* **CSY** - [csy0522](https://github.com/csy0522)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details


