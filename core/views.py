# core/views.py
import pandas as pd
from django.shortcuts import render
from .forms import DatasetForm
from .models import Dataset
from io import StringIO
import os
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import base64



def upload_dataset(request):
    dataset_info = None  

    if request.method == 'POST':
        form = DatasetForm(request.POST, request.FILES)
        if form.is_valid():
            dataset = form.save()
            file_path = dataset.file.path

            
            df = pd.read_csv(file_path)

            
            tmp_dir = os.path.join(os.getcwd(), 'tmp')  
            if not os.path.exists(tmp_dir):
                os.makedirs(tmp_dir)  

            # Save the dataset in the tmp directory
            temp_file_path = os.path.join(tmp_dir, 'uploaded_dataset.csv')
            df.to_csv(temp_file_path, index=False)
            request.session['dataset_path'] = temp_file_path

            # Extract dataset information for display
            buffer = StringIO()
            df.info(buf=buffer)
            info_output = buffer.getvalue()
            buffer.close()

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

            dataset_info = {
                'preview': df.head(15).to_html(classes='table table-striped table-hover', index=False),
                'info': column_info,
                'describe': df.describe().reset_index().to_html(classes='table table-striped table-hover', index=False),
                'shape': df.shape,
            }

    else:
        form = DatasetForm()

    return render(request, 'core/upload.html', {'form': form, 'dataset_info': dataset_info})




def visualize_data(request):
    dataset_path = request.session.get('dataset_path')  # Retrieve the uploaded dataset path
    if not dataset_path or not os.path.exists(dataset_path):
        return render(request, 'core/visualize.html', {'error': 'No dataset available for visualization. Please upload a dataset first.'})

    # Load the dataset
    df = pd.read_csv(dataset_path)
    plot_type = request.GET.get('plot_type', 'Histogram')  # Default plot type is 'Histogram'
    plot = None

    # Generate the requested plot
    if plot_type == 'Histogram':
        fig, ax = plt.subplots(figsize=(6, 4))
        numeric_columns = df.select_dtypes(include=['float', 'int']).columns
        if len(numeric_columns) > 0:
            sns.histplot(df[numeric_columns[0]], kde=True, ax=ax)  # Plot the first numeric column
            ax.set_title(f"Histogram of {numeric_columns[0]}")
            plot = _plot_to_base64(fig)

    elif plot_type == 'Heatmap':
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
        ax.set_title("Correlation Heatmap")
        plot = _plot_to_base64(fig)

    elif plot_type == 'Pointplot':
        fig, ax = plt.subplots(figsize=(6, 4))
        if len(df.columns) >= 2:
            sns.pointplot(x=df.columns[0], y=df.columns[1], data=df, ax=ax)
            ax.set_title(f"Point Plot of {df.columns[0]} vs {df.columns[1]}")
            plot = _plot_to_base64(fig)

    elif plot_type == 'Box Plot':
        fig, ax = plt.subplots(figsize=(8, 6))

    # Select numeric columns and exclude 'Id'
        numeric_columns = df.select_dtypes(include=['float', 'int']).columns.tolist()
        numeric_columns = [col for col in numeric_columns if col.lower() not in ['id']]

        if len(numeric_columns) > 0:
            # Plot only the relevant numeric columns
            sns.boxplot(data=df[numeric_columns], ax=ax)
            ax.set_title("Box Plot of Numeric Columns")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)  # Rotate labels for better readability
            plot = _plot_to_base64(fig)
        else:
            plot = None  # Handle cases with no numeric columns

    
    return render(request, 'core/visualize.html', {'plot': plot, 'plot_type': plot_type})


def _plot_to_base64(fig):
    """Convert Matplotlib figure to a base64 image for rendering in templates."""
    buffer = BytesIO()
    fig.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    buffer.close()
    plt.close(fig)  # Close the figure to save memory
    return image_base64


def home(request):
    return render(request, 'core/home.html')

def upload(request):
    return render(request, 'core/upload.html')

def visualize(request):
    return render(request, 'core/visualize.html')

def preprocess(request):
    return render(request, 'core/preprocess.html')

def classify(request):
    return render(request, 'core/classify.html')

def predict(request):
    return render(request, 'core/predict.html')
