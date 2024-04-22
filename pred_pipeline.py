from stroke.DataIngestion import kaggle_data_download, load_dataframe
from zenml.pipelines import pipeline
from stroke.DataProcessing import preprocess_data
from stroke.ModelTraining import model_trainer
from stroke.model_promoter import model_promoter
from stroke.inference_predict import inference_predict
from stroke.ModelEvaluvation import model_evaluator

@pipeline(enable_cache=True)
def first_pipeline(
    kaggle_data_download,
    load_dataframe,
    preprocess_data,
    model_trainer,
    model_evaluator,
    model_promoter,
    inference_predict
):
    path = kaggle_data_download()
    data = load_dataframe(path)
    X_train, X_test, y_train, y_test = preprocess_data(data)
    trained_model = model_trainer(X_train, y_train, model_type='knn')
    accuracy = model_evaluator(trained_model, X_train, X_test, y_train, y_test)
    is_inferenced = model_promoter(accuracy)
    y_pred = inference_predict(trained_model, X_test)

    print(y_pred)
    if is_inferenced:
        print("promoted")
    else:
        print("not promoted")


if __name__ == "__main__":
    first_pipeline(
        kaggle_data_download=kaggle_data_download(),
        load_dataframe=load_dataframe(),
        preprocess_data=preprocess_data(),
        model_trainer=model_trainer,
        model_evaluator=model_evaluator,
        model_promoter=model_promoter,
        inference_predict=inference_predict
    ).run()
