import matplotlib.pyplot as plt

def plot_strategy(time, Ih_control, Ih_comparison, label_control, label_comparison, title, ylabel):
    plt.figure(figsize=(10, 5))
    plt.plot(time, Ih_comparison, 'r-', label=label_comparison)
    plt.plot(time, Ih_control, 'b-', label=label_control)
    plt.xlabel('Time (weeks)')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()

def plot_controls(time, u1=None, u2=None, u3=None, title="Control Strategy Over Time"):
    plt.figure(figsize=(10, 5))
    
    if u1 is not None:
        plt.plot(time, u1, 'r-', linewidth=2, label='u1 (Larvicide)')
    if u2 is not None:
        plt.plot(time, u2, 'b-', linewidth=2, label='u2 (Fogging)')
    if u3 is not None:
        plt.plot(time, u3, 'g-', linewidth=2, label='u3 (Vaccination)')
        
    plt.xlabel('Time (weeks)')
    plt.ylabel('Control Variable Value')
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()