import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import plot_importance


def generate_plots(model, X_train_poly, y_train, training_data):
    plt.figure(figsize=(10, 6))
    plot_importance(model, max_num_features=10)
    plt.title('Feature Importance')
    plt.savefig('hackathon_code/feature_importance.png')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.scatter(y_train, model.predict(X_train_poly), alpha=0.5)
    plt.xlabel('Actual Passengers Up')
    plt.ylabel('Predicted Passengers Up')
    plt.title('Actual vs. Predicted Passengers Up')
    plt.savefig('hackathon_code/actual_vs_predicted.png')
    plt.show()

    plt.figure(figsize=(12, 8))
    corr = training_data[['line_id', 'direction', 'station_index', 'latitude', 'longitude']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix of Features')
    plt.savefig('hackathon_code/correlation_matrix_features.png')
    plt.show()

    feature = 'line_id'
    plt.figure(figsize=(10, 6))
    plt.scatter(training_data[feature], training_data['passengers_up'], alpha=0.5)
    plt.xlabel(feature)
    plt.ylabel('Passengers Up')
    plt.title(f'Scatter Plot of {feature} vs Passengers Up')
    plt.savefig(f'hackathon_code/{feature}_vs_passengers_up.png')
    plt.show()

   # Set the figure size to be smaller
    plt.figure(figsize=(8, 6))

    # Calculate the correlation matrix
    correlation_matrix = training_data.corr()

    # Create the heatmap with smaller annotations and reduced font sizes
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5,
                annot_kws={"size": 8}, cbar_kws={"shrink": 0.8})

    # Set the title with a smaller font size
    plt.title('Correlation Heatmap of Features', fontsize=10)

    # Adjust the tick parameters for better readability
    plt.xticks(fontsize=8, rotation=45, ha='right')
    plt.yticks(fontsize=8)

    # Add padding to move the plot higher
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust the rect parameter to add padding

    # Show the plot
    plt.show()
