import re
import matplotlib.pyplot as plt

global_steps = []
step_losses = []
epochs = []
avg_losses = []
avg_accs = []

epoch_counter = 1
steps_per_epoch = None

# Read log file and extract metrics
with open('C:/Users/Proprietario/repo/midiR1/logs/training.log', 'r') as f:
    for line in f:

        m_step = re.search(r"Step\s+(\d+)/(\d+)\s+---\s+Loss:\s+([0-9.]+)", line)
        if m_step:
            step_num = int(m_step.group(1))
            total_steps = int(m_step.group(2))
            loss = float(m_step.group(3))

            if steps_per_epoch is None:
                steps_per_epoch = total_steps

            global_step = (epoch_counter - 1) * steps_per_epoch + step_num
            global_steps.append(global_step)
            step_losses.append(loss)
            continue

        m_epoch = re.search(r"\[Epoch\s+(\d+)\]\s+avg loss:\s+([0-9.]+)\s+avg_acc:\s+([0-9.]+)", line)
        if m_epoch:
            epoch_idx = int(m_epoch.group(1))
            avg_loss = float(m_epoch.group(2))
            avg_acc = float(m_epoch.group(3))
            epochs.append(epoch_idx)
            avg_losses.append(avg_loss)
            avg_accs.append(avg_acc)
            epoch_counter = epoch_idx + 1

# Plot step-wise (global) loss
plt.figure()
plt.plot(global_steps, step_losses)
plt.xlabel('Global Step')
plt.ylabel('Loss')
plt.title('Training Loss per Global Step')
plt.grid(True)
plt.show()

plt.figure()
plt.plot(epochs, avg_losses)
plt.xlabel('Epoch')
plt.ylabel('Avg Loss')
plt.title('Training Loss per Epoch')
plt.grid(True)
plt.show()

# Plot epoch-wise average accuracy
plt.figure()
plt.plot(epochs, avg_accs)
plt.xlabel('Epoch')
plt.ylabel('Average Accuracy')
plt.title('Average Accuracy per Epoch')
# Set x-ticks every 20 epochs
if epochs:
    plt.xticks(range(min(epochs), max(epochs) + 1, 20))
plt.grid(True)
plt.show()

