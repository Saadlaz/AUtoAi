# core/views.py
import numpy as np
import pandas as pd
import base64
import os
import io
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from datetime import datetime
from django.shortcuts import render, redirect
from .forms import DatasetForm
from .models import Dataset 
from io import StringIO
from io import BytesIO
from sklearn.preprocessing import LabelEncoder , MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from django.http import HttpResponse
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from django.conf import settings
from .forms import DynamicDatasetForm
from django.utils import timezone




def upload_dataset(request):
    dataset_info = {}  # Dictionary to hold dataset details

    if request.method == 'POST':
        form = DatasetForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = form.cleaned_data['file']

            # Save uploaded file
            dataset = Dataset.objects.create(
                name=f"dataset_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}",
                file=uploaded_file
            )

            # Save dataset ID in session
            request.session['dataset_id'] = dataset.id

            # Load dataset and get information
            file_path = dataset.file.path
            df = pd.read_csv(file_path)

            # Use the helper function to get dataset information
            dataset_info = _get_dataset_info(df)

    else:
        # Handle dataset reload if ID exists in session
        form = DatasetForm()
        dataset_id = request.session.get('dataset_id')
        if dataset_id:
            try:
                dataset = Dataset.objects.get(id=dataset_id)
                file_path = dataset.file.path
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    dataset_info = _get_dataset_info(df)
            except Dataset.DoesNotExist:
                # Clear session if dataset doesn't exist
                del request.session['dataset_id']

    return render(request, 'core/upload.html', {'form': form, 'dataset_info': dataset_info})


def _get_dataset_info(df):
    """Helper function to extract dataset information."""
    buffer = StringIO()
    df.info(buf=buffer)
    info_output = buffer.getvalue()
    buffer.close()

    # Process Pandas DataFrame info output
    column_info = []
    for line in info_output.splitlines()[5:]:
        if "dtypes:" in line or "memory usage:" in line:
            continue
        parts = line.split(maxsplit=4)
        if len(parts) == 5:
            column_info.append({
                'Index': parts[0],
                'ColumnName': parts[1],
                'NonNullCount': parts[2],
                'DataType': parts[4],
            })

    return {
        'preview': df.head(10).to_html(classes='table table-striped table-hover', index=False),
        'info': column_info,
        'describe': df.describe().reset_index().to_html(classes='table table-striped table-hover', index=False),
        'shape': df.shape,
    }


def encode_data(request):
    dataset_id = request.session.get('dataset_id')
    if not dataset_id:
        return render(request, 'core/upload.html', {'error': 'No dataset available for encoding. Please upload a dataset first.'})

    # Fetch and load the dataset
    try:
        dataset = Dataset.objects.get(id=dataset_id)
        file_path = dataset.file.path
        df = pd.read_csv(file_path)
    except Exception as e:
        return render(request, 'core/upload.html', {'error': f"Error loading dataset: {str(e)}"})

    # Store mappings for each categorical column
    label_mappings = {}

    # Encode categorical features
    try:
        categorical_columns = df.select_dtypes(include=['object']).columns
        if not categorical_columns.empty:
            label_encoder = LabelEncoder()
            for column in categorical_columns:
                # Store the mapping
                label_mappings[column] = dict(zip(label_encoder.fit_transform(df[column]), df[column]))
                # Encode the feature
                df[column] = label_encoder.transform(df[column])

            # Save the encoded data
            df.to_csv(file_path, index=False)

            # Save the mapping to the session (or a database if needed)
            request.session['label_mappings'] = label_mappings
        else:
            return render(request, 'core/upload.html', {'error': 'No categorical columns to encode.'})
    except Exception as e:
        return render(request, 'core/upload.html', {'error': f"Error encoding dataset: {str(e)}"})

    return render(request, 'core/upload.html', {'message': 'Dataset successfully encoded and saved.'})



def visualize_data(request):
    # Load dataset
    dataset_id = request.session.get('dataset_id')
    if not dataset_id:
        return render(request, 'core/visualize.html', {'error': 'No dataset available. Please upload one.'})

    try:
        dataset = Dataset.objects.get(id=dataset_id)
        file_path = dataset.file.path
        df = pd.read_csv(file_path)
    except Exception as e:
        return render(request, 'core/visualize.html', {'error': f"Error loading dataset: {str(e)}"})

    # Extract column names
    columns = df.columns.tolist()
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

    # Get user inputs
    plot_type = request.GET.get('plot_type', 'Heatmap')
    x_label = request.GET.get('x_label', None)
    y_label = request.GET.get('y_label', None)

    plot_html = None

    try:
        # If Heatmap is selected, encode categorical data
        if plot_type == 'Heatmap':
            if categorical_columns:
                label_encoder = LabelEncoder()
                for column in categorical_columns:
                    df[column] = label_encoder.fit_transform(df[column])

            # Generate the Heatmap with a custom color scale and size
            corr = df.corr()
            fig = px.imshow(
                corr, 
                text_auto=True, 
                title="Correlation Heatmap (Encoded Data)", 
                width=1200,  # Width
                height=900,  # Height
                color_continuous_scale='purpor'  # Use purpor color scale
            )
            plot_html = fig.to_html(full_html=False)

        elif plot_type == 'Histogram' and x_label:
            fig = px.histogram(df, x=x_label, title=f"Histogram of {x_label}")
            plot_html = fig.to_html(full_html=False)

        elif plot_type == 'Scatter' and x_label and y_label:
            fig = px.scatter(
                df, 
                x=x_label, 
                y=y_label, 
                title=f"Scatter Plot: {x_label} vs {y_label}",
                color_continuous_scale='purpor'  # Use purpor for point colors
            )
            plot_html = fig.to_html(full_html=False)

        elif plot_type == 'Pair Plot':
            fig = px.scatter_matrix(
                df, 
                dimensions=numerical_columns, 
                title="Pair Plot of Numerical Features",
                color_continuous_scale='purpor'  # Use purpor color scale
            )
            plot_html = fig.to_html(full_html=False)

        elif plot_type == 'Pie' and x_label:
            fig = px.pie(
                df, 
                names=x_label,
                title=f"Pie Chart of {x_label}",
                color_discrete_sequence=px.colors.sequential.Purpor  # Use purpor for pie slices
            )
            plot_html = fig.to_html(full_html=False)

        elif plot_type == 'Violin Plot' and x_label and y_label:
            fig = px.violin(df, x=x_label, y=y_label, box=True, title=f"Violin Plot of {y_label} by {x_label}")
            plot_html = fig.to_html(full_html=False)


    except Exception as e:
        return render(request, 'core/visualize.html', {'error': f"Error generating plot: {str(e)}"})

    return render(request, 'core/visualize.html', {
        'plot_html': plot_html,
        'columns': columns,
        'plot_type': plot_type,
        'x_label': x_label,
        'y_label': y_label,
    })





#-------------------------------------------------------






def create_dynamic_dataset(request):
    if request.method == 'POST':
        form = DynamicDatasetForm(request.POST)
        if form.is_valid():
            column_names = [name.strip() for name in form.cleaned_data['column_names'].split(',')]
            column_types = [typ.strip().lower() for typ in form.cleaned_data['column_types'].split(',')]
            num_rows = form.cleaned_data['num_rows']

            # Generate Data Dynamically
            data = {}
            for col_name, col_type in zip(column_names, column_types):
                if col_type == 'int':
                    data[col_name] = np.random.randint(0, 100, num_rows)
                elif col_type == 'float':
                    data[col_name] = np.random.uniform(0, 100, num_rows)
                elif col_type == 'str':
                    data[col_name] = [f"{col_name}_{i}" for i in range(num_rows)]
                else:
                    data[col_name] = ['N/A'] * num_rows  # Default to N/A for unknown types
            
            # Convert to DataFrame
            df = pd.DataFrame(data)

            # Save CSV File
            file_name = f"dynamic_dataset_{timezone.now().strftime('%Y%m%d_%H%M%S')}.csv"
            file_path = os.path.join(settings.MEDIA_ROOT, file_name)
            df.to_csv(file_path, index=False)

            # Save to Database
            Dataset.objects.create(
                name=file_name,
                file=file_name
            )

            # Redirect to upload page
            return redirect('upload_dataset')
    else:
        form = DynamicDatasetForm()

    return render(request, 'core/create_dynamic_dataset.html', {'form': form})



def home(request):
    return render(request, 'core/home.html')

def upload(request):
    return render(request, 'core/upload.html')


def preprocess(request):
    return render(request, 'core/preprocess.html')

def classify(request):
    return render(request, 'core/classify.html')

def predict(request):
    return render(request, 'core/predict.html')






#---------------------- Preprocessing ---------------------------------






def preprocess(request):
    df = None
    missing_summary, train_shape, test_shape, sample_data = None, None, None, None
    dataset_id = request.session.get('dataset_id')
    
    if not dataset_id:
        return render(request, 'core/upload.html', {'error': 'No dataset available for preprocessing. Please upload a dataset first.'})

    try:
        dataset = Dataset.objects.get(id=dataset_id)
        file_path = dataset.file.path
        df = pd.read_csv(file_path)
    except Exception as e:
        return render(request, 'core/upload.html', {'error': f"Error loading dataset: {str(e)}"})

    # Initialize variables
    num_duplicates_deleted = 0
    encoded_columns = []
    num_outliers_deleted = 0
    
    if df is not None:
        # Handle Missing Values
        missing_summary = df.isnull().sum().to_dict()
        for column in df.columns:
            if df[column].isnull().sum() > 0:
                if df[column].dtype == 'object':  # Categorical columns
                    df[column].fillna("Unknown", inplace=True)
                else:  # Numerical columns
                    df[column].fillna(df[column].median(), inplace=True)

        # Remove Duplicates
        num_duplicates_deleted = df.duplicated().sum()
        df.drop_duplicates(inplace=True)

        # Strip Whitespaces
        df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

        # Encoding Categorical Features
        for col in df.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoded_columns.append(col)

        # Outlier Detection with IQR Method
        for col in df.select_dtypes(include=['float64', 'int64']).columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_limit = Q1 - 1.5 * IQR
            upper_limit = Q3 + 1.5 * IQR

            # Identify outliers for the column
            outliers = ((df[col] > upper_limit) | (df[col] < lower_limit))
            num_outliers_deleted += outliers.sum()

            # Clip the column to the limits
            df[col] = df[col].clip(lower=lower_limit, upper=upper_limit)


        # Scaling & Normalization
        scaler = MinMaxScaler()
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

        # Save preprocessed data as a CSV file in the 'datasets' folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        preprocessed_file_name = f'preprocessed_{timestamp}.csv'
        datasets_folder = os.path.join(settings.BASE_DIR, 'datasets')
        os.makedirs(datasets_folder, exist_ok=True)
        preprocessed_file_path = os.path.join(datasets_folder, preprocessed_file_name)
        df.to_csv(preprocessed_file_path, index=False)

        # Store file path in session
        request.session['preprocessed_file'] = preprocessed_file_path

        # Train-Test Split (to show shape)
        train, test = train_test_split(df, test_size=0.2, random_state=42)
        train_shape, test_shape = train.shape, test.shape

        # Prepare Data Preview
        sample_data = df.head(10).to_html(classes="table table-bordered")

    # Context for rendering
    context = {
        'missing_summary': missing_summary,
        'train_shape': train_shape,
        'test_shape': test_shape,
        'sample_data': sample_data,
        'num_duplicates_deleted': num_duplicates_deleted,
        'encoded_columns': encoded_columns,
        'num_outliers_deleted': num_outliers_deleted,
    }
    return render(request, 'core/preprocess.html', context)






#----------------------------- Model Training ----------------------------------






from sklearn.cluster import KMeans, DBSCAN
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, f1_score,
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    mean_absolute_error, mean_squared_error, r2_score
)
import plotly.figure_factory as ff
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA

def classify(request):
    kmeans_plot_html = None
    dbscan_plot_html = None
    classification_plots_html = []
    regression_plots_html = []    
    result_list = []
    models_by_category = {
        "Classification": ['Naive Bayes', 'SVM', 'Random Forest', 'K-Nearest Neighbours', 'Artificial Neural Network', 'Logistic Regression'],
        "Clustering": ['K-Means', 'DBSCAN'],
        "Regression": ['Simple', 'Polynomial', 'Multiple']
    }
    metrics_by_category = {
        "Classification": ["Accuracy", "Precision", "F1"],
        "Clustering": ["Silhouette Score", "Calinski Harabasz", "Davies-Bouldin"],
        "Regression": ["MAE", "MSE", "R2"]
    }

    # Load preprocessed data file from session
    preprocessed_file_path = request.session.get('preprocessed_file')
    if not preprocessed_file_path:
        return render(request, 'core/preprocess.html', {'error': 'No preprocessed data available.'})

    # Load the dataset
    df = pd.read_csv(preprocessed_file_path)
    column_names = df.columns.tolist()

    # Get selected ML category
    ml_category = request.POST.get('ml_category', 'Classification')
    available_models = models_by_category.get(ml_category, [])
    selected_models = request.POST.getlist('models')
    
    try:
        if ml_category == "Classification":
            target_column = request.POST.get('target_column', column_names[-1])
            X = df.drop(columns=[target_column])
            y = df[target_column]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # List to store confusion matrices for each model
            classification_plots_html = []
            
            for model_name in selected_models:
                if model_name == "Naive Bayes":
                    model = GaussianNB()
                elif model_name == "SVM":
                    model = SVC()
                elif model_name == "Random Forest":
                    model = RandomForestClassifier()
                elif model_name == "K-Nearest Neighbours":
                    model = KNeighborsClassifier()
                elif model_name == "Artificial Neural Network":
                    model = MLPClassifier(max_iter=500)
                elif model_name == "Logistic Regression":
                    model = LogisticRegression(max_iter=1000)
                else:
                    continue

                # Train the model and make predictions
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # Calculate classification metrics
                metrics = {
                    "Accuracy": accuracy_score(y_test, y_pred) * 100,
                    "Precision": precision_score(y_test, y_pred, average='weighted') * 100,
                    "F1": f1_score(y_test, y_pred, average='weighted') * 100
                }
                result_list.append({"model_name": model_name, "metrics": metrics})

                # Generate Confusion Matrix for the current model
                confusion = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
                confusion_fig = ff.create_annotated_heatmap(
                    z=confusion.values,
                    x=list(confusion.columns),
                    y=list(confusion.index),
                    colorscale='purpor',
                    showscale=True
                )
                confusion_fig.update_layout(
                    title_text=f"Confusion Matrix ({model_name})",
                    xaxis_title="Predicted",
                    yaxis_title="Actual"
                )
                classification_plots_html.append(confusion_fig.to_html(full_html=False))


        elif ml_category == "Clustering":
            X = df  # Use the entire dataset as features for clustering

            # Initialize cluster visualizations
            kmeans_plot_html = None
            dbscan_plot_html = None

            for model_name in selected_models:
                if model_name == "K-Means":
                    model = KMeans(n_clusters=3, random_state=42)  # Example: 3 clusters
                    y_pred = model.fit_predict(X)

                    # Calculate clustering metrics (unsupervised only)
                    metrics = {
                        "Silhouette Score": silhouette_score(X, y_pred) * 100,
                        "Calinski Harabasz": calinski_harabasz_score(X, y_pred),
                        "Davies-Bouldin": davies_bouldin_score(X, y_pred),
                    }
                    result_list.append({"model_name": model_name, "metrics": metrics})

                    # Interactive visualization for K-Means using Plotly
                    pca = PCA(n_components=2)
                    X_pca = pca.fit_transform(X)
                    kmeans_fig = px.scatter(
                        x=X_pca[:, 0],
                        y=X_pca[:, 1],
                        color=y_pred.astype(str),
                        title="K-Means Clustering",
                        labels={"x": "PCA 1", "y": "PCA 2"},
                    )
                    kmeans_plot_html = kmeans_fig.to_html(full_html=False)

                elif model_name == "DBSCAN":
                    model = DBSCAN(eps=0.5, min_samples=5)  # Default values
                    y_pred = model.fit_predict(X)

                    # Check if valid clusters exist (excluding noise points)
                    unique_labels = set(y_pred)
                    n_clusters = len(unique_labels - {-1})  # Exclude noise points (-1)

                    if n_clusters > 0:
                        # Calculate clustering metrics
                        metrics = {
                            "Silhouette Score": silhouette_score(X, y_pred) * 100 if n_clusters > 1 else "N/A",
                            "Calinski Harabasz": calinski_harabasz_score(X, y_pred),
                            "Davies-Bouldin": davies_bouldin_score(X, y_pred),
                        }
                    else:
                        # No valid clusters
                        metrics = {
                            "Silhouette Score": "N/A",
                            "Calinski Harabasz": "N/A",
                            "Davies-Bouldin": "N/A",
                        }

                    result_list.append({"model_name": model_name, "metrics": metrics})

                    # Generate Interactive Visualization for DBSCAN
                    pca = PCA(n_components=2)
                    X_pca = pca.fit_transform(X)
                    dbscan_fig = px.scatter(
                        x=X_pca[:, 0],
                        y=X_pca[:, 1],
                        color=y_pred.astype(str),
                        title="DBSCAN Clustering",
                        labels={"x": "PCA 1", "y": "PCA 2"},
                        color_discrete_map={"-1": "red"}  # Assign red for noise points (-1)
                    )
                    dbscan_fig.update_layout(legend_title_text="Cluster")
                    dbscan_plot_html = dbscan_fig.to_html(full_html=False)



        elif ml_category == "Regression":
            # Ensure the target column is selected
            target_column = request.POST.get('target_column', column_names[-1])
            X = df.drop(columns=[target_column])  # Features
            y = df[target_column]  # Target

            # Split data into training and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # List to store regression plots for all models
            regression_plots_html = []

            for model_name in selected_models:
                if model_name == "Simple":
                    model = LinearRegression()
                elif model_name == "Polynomial":
                    best_r2 = float('-inf')
                    best_degree = 1

                    for degree in range(1, 5):  # Test degrees 1 to 4
                        poly = PolynomialFeatures(degree=degree)
                        X_poly_train = poly.fit_transform(X_train)
                        X_poly_test = poly.transform(X_test)

                        temp_model = LinearRegression()
                        temp_model.fit(X_poly_train, y_train)
                        temp_y_pred = temp_model.predict(X_poly_test)

                        temp_r2 = r2_score(y_test, temp_y_pred) * 100

                        if temp_r2 > best_r2:
                            best_r2 = temp_r2
                            best_degree = degree
                            model = temp_model
                            y_pred = temp_y_pred

                    metrics = {
                        "Best Degree": best_degree,
                        "MAE": mean_absolute_error(y_test, y_pred),
                        "MSE": mean_squared_error(y_test, y_pred),
                        "R2": best_r2
                    }

                    result_list.append({"model_name": f"Polynomial (Degree {best_degree})", "metrics": metrics})
                    continue

                elif model_name == "Multiple":
                    model = LinearRegression()
                else:
                    continue

                # Train and predict
                if model_name != "Polynomial":
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                # Calculate regression metrics
                metrics = {
                    "MAE": mean_absolute_error(y_test, y_pred),
                    "MSE": mean_squared_error(y_test, y_pred),
                    "R2": r2_score(y_test, y_pred) * 100
                }
                result_list.append({"model_name": model_name, "metrics": metrics})

                # Generate Actual vs Predicted Plot for the current model
                regression_fig = px.scatter(
                    x=y_test,
                    y=y_pred,
                    labels={'x': "Actual Values", 'y': "Predicted Values"},
                    title=f"Actual vs Predicted ({model_name})"
                )
                regression_fig.add_shape(type="line", x0=y_test.min(), y0=y_test.min(),
                                        x1=y_test.max(), y1=y_test.max(), line=dict(color="red", dash="dash"))
                regression_plots_html.append(regression_fig.to_html(full_html=False))


    except Exception as e:
        return render(request, 'core/classify.html', {
            'error': f"Error occurred: {str(e)}",
            'models': available_models,
            'columns': column_names,
            'ml_category': ml_category
        })

    context = {
    "columns": column_names,
    "available_models": available_models,
    "ml_category": ml_category,
    "results": result_list,
    "kmeans_plot_html": kmeans_plot_html,
    "dbscan_plot_html": dbscan_plot_html,
    "classification_plots_html": classification_plots_html,
    "regression_plots_html": regression_plots_html, 
}
    return render(request, 'core/classify.html', context)