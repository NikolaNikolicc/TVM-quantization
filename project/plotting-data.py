
import ast
import numpy as np
import matplotlib.pyplot as plt

def read_file(file_path):
    # Read file
    with open(file_path, 'r') as file:
        data_lines = file.readlines()

    # Convert lines to dict
    data_lines = [ast.literal_eval(line.split(':', 1)[1].strip()) for line in data_lines]
    return data_lines

def process_and_compute_means(data_lines):
    # Prepare dict
    sums = {
        "unquantized": {True: [], False: []},
        "quantized8": {True: [], False: []},
        "quantized16": {True: [], False: []},
        "quantized32": {True: [], False: []}
    }

    # Convert lines to dict
    for iteration_data in data_lines:
        # iteration_data = ast.literal_eval(line.split(':', 1)[1].strip())

        for key, value in iteration_data.items():
            sums[key][True].append(value[True][0])
            sums[key][False].append(value[False][0])

    # Calculate mean values and return result
    means = {key: {k: np.mean(v) for k, v in value.items()} for key, value in sums.items()}
    return means

def plotting_mean_values(computed_means):
    # Preparing data for the bar chart
    categories = list(computed_means.keys())
    true_values = [computed_means[category][True] for category in categories]
    false_values = [computed_means[category][False] for category in categories]
    categories = ["Unquantized", "Quantized to int8(int8)", "Quantized to int8(int16)", "Quantized to int8(int32)"]

    # Creating the bar chart
    fig, ax = plt.subplots()

    # Bar setup
    bar_width = 0.35
    index = np.arange(len(categories))
    bar1 = ax.bar(index, true_values, bar_width, label='GPU', color='skyblue')
    bar2 = ax.bar(index + bar_width, false_values, bar_width, label='CPU', color='sandybrown')

    # Legend setup
    ax.set_xlabel('Type of compilation')
    ax.set_ylabel('Time [milliseconds]')
    ax.set_title('Average TVM Compilation Time after 450 executions')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(categories)
    ax.legend(loc="upper left", bbox_to_anchor=(1,1), title="Target device")

    # Adding grid lines for better readability of the values
    plt.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5)

    # Add text labels beside the bars
    for rect in bar1 + bar2:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., 1.005 * height,
                f'{height:.2f}', ha='center', va='bottom')

    # Adding note
    plt.text(0.5, -0.15, "Note: In the 'Type of Compilation', the values in brackets represent the 'activation_dtype' parameter.",
             fontsize=10, ha='center', va='top', transform=ax.transAxes, color='gray')


    # Improve the layout to prevent clipping of tick-labels
    plt.tight_layout()

    # Get the figure manager (display settings)
    mng = plt.get_current_fig_manager()

    # Set window size: (width, height) in pixels
    window_width = 1500
    window_height = 600
    mng.resize(window_width, window_height)

    fig.set_size_inches(window_width / fig.dpi, window_height / fig.dpi)
    plt.savefig('plt_avg_tvm_compilation_time.png')
    plt.show()

def plotting_all_values_gpu(data_lines):
    # Extracting the True values for each category
    unquantized = [item['unquantized'][True][0] for item in data_lines]
    quantized8 = [item['quantized8'][True][0] for item in data_lines]
    quantized16 = [item['quantized16'][True][0] for item in data_lines]
    quantized32 = [item['quantized32'][True][0] for item in data_lines]

    num_iterations = len(unquantized)

    # Creating the scatter plot with lines
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_iterations + 1), unquantized, label='Unquantized', marker='o')
    plt.plot(range(1, num_iterations + 1), quantized8, label='Quantized to int8(int8)', marker='o')
    plt.plot(range(1, num_iterations + 1), quantized16, label='Quantized to int8(int16)', marker='o')
    plt.plot(range(1, num_iterations + 1), quantized32, label='Quantized to int8(int32)', marker='o')

    plt.xticks(range(1, num_iterations + 1))

    # Adding labels and title
    plt.xlabel('Iterations (each iteration represents 30 executions)')
    plt.ylabel('Time [milliseconds]')
    plt.title('Performance Results by Iteration for TVM Compilation on GPU')
    plt.legend(title="Type of compilation")

    # Adding note
    plt.text(0.5, -0.15,
             "Note: In the 'Type of Compilation' box, the values in brackets represent the 'activation_dtype' parameter.",
             fontsize=10, ha='center', va='top', transform=plt.gca().transAxes, color='gray')

    # Improve the layout to prevent clipping of tick-labels
    plt.tight_layout()

    plt.savefig("plt_performance_results_all_iterations_gpu.png")
    # Show the plot
    plt.show()


def plotting_all_values_cpu(data_lines):
    # Extracting the True values for each category
    unquantized = [item['unquantized'][False][0] for item in data_lines]
    quantized8 = [item['quantized8'][False][0] for item in data_lines]
    quantized16 = [item['quantized16'][False][0] for item in data_lines]
    quantized32 = [item['quantized32'][False][0] for item in data_lines]

    num_iterations = len(unquantized)

    # Creating the scatter plot with lines
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_iterations + 1), unquantized, label='Unquantized', marker='o')
    plt.plot(range(1, num_iterations + 1), quantized8, label='Quantized to int8(int8)', marker='o')
    plt.plot(range(1, num_iterations + 1), quantized16, label='Quantized to int8(int16)', marker='o')
    plt.plot(range(1, num_iterations + 1), quantized32, label='Quantized to int8(int32)', marker='o')

    plt.xticks(range(1, num_iterations + 1))

    # Adding labels and title
    plt.xlabel('Iterations (each iteration represents 30 executions)')
    plt.ylabel('Time [milliseconds]')
    plt.title('Performance Results by Iteration for TVM Compilation on CPU')
    plt.legend(title="Type of compilation")

    # Adding note
    plt.text(0.5, -0.15,
             "Note: In the 'Type of Compilation' box, the values in brackets represent the 'activation_dtype' parameter.",
             fontsize=10, ha='center', va='top', transform=plt.gca().transAxes, color='gray')

    # Improve the layout to prevent clipping of tick-labels
    plt.tight_layout()

    plt.savefig("plt_performance_results_all_iterations_cpu.png")
    # Show the plot
    plt.show()

def main():
    # Read file and calculate mean values
    file_path = 'output-for-plotting.txt'
    data_lines = read_file(file_path)
    print(data_lines)
    computed_means = process_and_compute_means(data_lines)
    # print(computed_means)
    plotting_mean_values(computed_means)
    plotting_all_values_gpu(data_lines)
    plotting_all_values_cpu(data_lines)

main()
