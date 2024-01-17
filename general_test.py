import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import metrics


def feature_selection(data, n_features=1000):
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(data['Statement'], data['Label'], test_size=0.2,
                                                        random_state=42)

    # 定义特征提取和分类器
    text_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('chi2', SelectKBest(chi2, k=n_features)),
        ('clf', MultinomialNB())
    ])

    # 训练模型
    text_clf.fit(X_train, y_train)

    # 预测
    predicted = text_clf.predict(X_test)

    # 输出模型性能
    print(metrics.classification_report(y_test, predicted))

    # 输出特征选择的结果
    feature_names = text_clf.named_steps['vect'].get_feature_names_out()
    selected_features = [feature_names[i] for i in text_clf.named_steps['chi2'].get_support(indices=True)]

    print(f"Selected Features: {selected_features}")

    return selected_features


# 读取数据
data = pd.read_csv('test.csv')  # 请替换为你的数据集路径

# 调用特征选择函数
selected_features = feature_selection(data)

