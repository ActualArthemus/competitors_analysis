import matplotlib.pyplot as plt
import pandas as pd
import nltk
import seaborn as sns
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


nltk.download('punkt')
pd.set_option('display.max_columns', None)


def draw_prices(dataset):
    """Draw a graph with prices"""
    plt.hist(dataset["Selling price"], label="Цена")
    plt.xlabel("Цена")
    plt.ylabel("Количество смартфонов")
    plt.title('Ценовой диапазон')
    plt.legend()
    plt.show()


def draw_average_prices(company, price):
    """Draw a graph with average prices per company"""
    plt.bar(company, price, label='Цена')
    plt.xlabel('Компании')
    plt.ylabel('Средняя цена')
    plt.title('Средняя цена смартфонов')
    plt.legend()
    plt.show()


def resolve_company_names(companies):
    """
    This method finds the names of companies.

    Args:
        names: contains the names of companies

    Returns:
        Sorted list of names
        """

    names = list()
    for mobilePhone in companies:
        names.append(word_tokenize(mobilePhone)[0])
    names = set(names)
    return names


def count_average_price(companies, dataset):
    """
    This method finds the average price per phone for each company.

    Args:
        average_price: contains the average price of smartphones for each manufacturer
        filtered_ds: contains the modified dataset
        price: contains averaged prices

    Returns:
        The average price of a smartphone for each manufacturer
        """
    average_price = list()
    for company in companies:
        filtered_ds = dataset[dataset['Mobile'].str.startswith(company)]
        price = filtered_ds['Selling price'].mean()
        average_price.append(price)
    return average_price


def split_data(dt, tg):
    """
    This method splits the data for test and train with .

    Args:
        dt: data set
        tg: name of target column ( for wich we find regression)

    Returns:
        f_train (feature train set), f_test (feature test set), t_train (target train set), t_test (target test set)
        """

    d = dt.drop("Mobile", axis=1)  # we drop this coz this column is name of phone and we dont use it in regression
    f = d.drop(tg, axis=1)  # feature data set is set of all data except target column
    td = d[tg]  # target data set is set of only target column
    f_train, f_test, t_train, t_test = train_test_split(f, td, test_size=0.25, random_state=42)
    return [f_train, f_test, t_train, t_test]


def fit_and_predict(fr1, fe1, tr1, te1, name):
    """
    This method fit data and
    train linear regression model then we
    print some results of linear regression and then
    draw plot of pairwise relationship in a whole dataset.

    Args:
        first 4 args is result of execution of train_test_split function
        fr1: feature set
        name: name of column for wich we build our regression (selling price/original price)

    Returns:
        f_train (feature train set), f_test (feature test set), t_train (target train set), t_test (target test set)
        """
    m = LinearRegression()  # create model for linear regression
    m.fit(fr1, tr1)  # fit model
    result = m.predict(fe1)  # predict
    # print some results of linear regression below
    print(f"{name} linear regression coeffs: \n", m.coef_)
    print(f"{name} Mean squared error: %.2f" % mean_squared_error(te1, result))
    print(f"{name} coeff of determination: %.2f" % r2_score(te1, result))

    # seaborn draw a plot of pairwise relationship in whole dataset
    train_dataset = fr1.copy()
    train_dataset.insert(0, "PRICE", tr1)
    _ = sns.pairplot(train_dataset, kind="reg", diag_kind="kde")

    return result


def main():
    dataset = pd.read_excel('datasetpython.xlsx', engine='openpyxl')
    dataset = dataset.dropna()
    mobile = list(dataset["Mobile"])

    company = list(resolve_company_names(mobile))

    price = list(count_average_price(company, dataset))

    draw_prices(dataset)
    draw_average_prices(company, price)

    origPrice = split_data(dataset, "Original price")
    sellingPrice = split_data(dataset, "Selling price")

    origResult = fit_and_predict(origPrice[0], origPrice[1], origPrice[2], origPrice[3], "original price")
    sellingResult = fit_and_predict(sellingPrice[0], sellingPrice[1], sellingPrice[2], sellingPrice[3], "selling price")


main()
