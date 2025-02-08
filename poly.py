import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter, LogFormatter, FuncFormatter, LogLocator
import argparse
import sys

def setup_parser():
    parser = argparse.ArgumentParser(
        description='Plot polynomial learning rate decay curves',
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument('-p', '--powers', type=float, nargs='+',
                      default=[0.1, 0.3, 0.5, 0.8, 1, 1.5, 2, 3, 5, 10],
                      help='Space-separated polynomial powers (e.g., 0.5 1 2)\n' +
                           'Default: 0.5 0.8 1 1.5 2 3')
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-4,
                      help='Initial learning rate (default: 1e-4)')
    parser.add_argument('-lre', '--lr-end', type=float, default=1e-7,
                      help='Final learning rate (default: 1e-7)')
    parser.add_argument('-s', '--steps', type=int, default=4000,
                      help='Total training steps (default: 4000)')
    parser.add_argument('-w', '--warmup', type=int, default=400,
                      help='Number of warmup steps (default: 50)')
    parser.add_argument('-o', '--output', type=str, default='lr_schedule.jpg',
                      help='Output image filename (default: lr_schedule.jpg)')
    parser.add_argument('--dpi', type=int, default=300,
                      help='Image DPI (default: 300)')
    parser.add_argument('-l', '--log-scale', type=str, choices=['none', 'standard', 'fine', 'wide'],
                      default='none',
                      help='''Logarithmic scale preset:
none: Linear scale (default)
standard: Regular log scale
fine: Dense log scale for small changes
wide: Extended log scale for large ranges''')
    parser.add_argument('-n', '--notation', type=str, choices=['s', 'd'], 
                      default='s',
                      help='''Y-axis notation:
s: Scientific notation (default)
d: Decimal notation''')
    
    return parser

def calculate_lr_schedule(t, total_steps, init_lr, final_lr, power, warmup_steps):
    """Calculate learning rate schedule with optional warmup"""
    if t < warmup_steps:
        # Linear warmup
        return init_lr * (t / warmup_steps)
    else:
        # Polynomial decay from init_lr to final_lr
        t_adjusted = t - warmup_steps
        total_steps_adjusted = total_steps - warmup_steps
        progress = t_adjusted / total_steps_adjusted
        if progress >= 1.0:
            return final_lr
        decay_factor = (1 - progress)**power
        lr_range = init_lr - final_lr
        return final_lr + lr_range * decay_factor

def setup_log_scale(ax, args, scale_type):
    """Configure logarithmic scale based on preset"""
    if scale_type == 'none':
        # Generate evenly spaced ticks
        num_ticks = 6
        y_ticks = np.linspace(args.lr_end, args.learning_rate, num_ticks)
        ax.set_yscale('linear')

        # Force scientific notation for linear scale if specified
        if args.notation == 's':
            formatter = FuncFormatter(lambda x, p: f'{x:.0e}')
            ax.yaxis.set_major_formatter(formatter)
            #ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))  #This one was not working correctly
            ax.yaxis.set_minor_formatter(plt.NullFormatter()) #removing the minor ticks
        else:
            ax.ticklabel_format(style='plain', axis='y')
        
        plt.yticks(y_ticks)
        plt.ylim(args.lr_end * 0.95, args.learning_rate * 1.05)
    else:
        ax.set_yscale('log')
        
        # Force scientific notation for log scale
        formatter = FuncFormatter(lambda x, p: f'{x:.0e}')
        ax.yaxis.set_major_formatter(formatter)
        
        if scale_type == 'standard':
            locmaj = LogLocator(base=10.0, numticks=6)
            ax.yaxis.set_major_locator(locmaj)
        elif scale_type == 'fine':
            locmaj = LogLocator(base=10.0, numticks=8)
            ax.yaxis.set_major_locator(locmaj)
            locmin = LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=8)
            ax.yaxis.set_minor_locator(locmin)
            ax.yaxis.set_minor_formatter(plt.NullFormatter())
        elif scale_type == 'wide':
            locmaj = LogLocator(base=10.0, numticks=6)
            ax.yaxis.set_major_locator(locmaj)
        
        plt.ylim(args.lr_end * 0.8, args.learning_rate * 1.2)

def generate_filename(args):
    """Generate filename from parameters in order of importance"""
    powers_str = 'p' + '-'.join([str(p) for p in args.powers])
    lr_str = f'lr{args.learning_rate:.0e}'
    lr_end_str = f'lre{args.lr_end:.0e}'
    steps_str = f's{args.steps}'
    warmup_str = f'w{args.warmup}'
    scale_str = f'scale_{args.log_scale}'
    notation_str = f'not_{args.notation}'
    dpi_str = f'dpi{args.dpi}'
    
    filename = f"lr_decay_{powers_str}_{lr_str}_{lr_end_str}_{steps_str}_{warmup_str}_{scale_str}_{notation_str}_{dpi_str}.jpg"
    return filename

def plot_schedules(args):
    """Generate and save the learning rate schedule plot"""
    # Generate filename if not explicitly provided
    if args.output == 'lr_schedule.jpg':
        args.output = generate_filename(args)
    
    # Use style sheet for better scientific notation
    plt.style.use('classic')
    
    # Set up figure
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    
    # Plot each power
    colors = ['purple', 'blue', 'cyan', 'turquoise', 'lightgreen', 'palegoldenrod', 'sandybrown', 'coral', 'orangered', 'red']
    
    # Generate time steps
    t = np.linspace(0, args.steps, 1000)
    
    # Plot each power
    for i, power in enumerate(args.powers):
        lr_schedule = [calculate_lr_schedule(step, args.steps, args.learning_rate, 
                                          args.lr_end, power, args.warmup) 
                      for step in t]
        
        if i < len(colors):
            color = colors[i]
        else:
            color = plt.cm.rainbow(i / len(args.powers))

        plt.plot(t, lr_schedule, label=f'power={power}', color=color, linewidth=2)
    
    # Set up scale based on user choice
    setup_log_scale(ax, args, args.log_scale)
    
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Learning Rate (η)', fontsize=12)
    title = f'Polynomial Learning Rate Decay ({args.log_scale.capitalize()} Scale)\nWarmup Steps: {args.warmup}'
    plt.title(title, fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Set x-axis ticks
    plt.xticks(np.linspace(0, args.steps, 5), 
              labels=[f'{int(x)}' for x in np.linspace(0, args.steps, 5)])
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(args.output, dpi=args.dpi, bbox_inches='tight', pad_inches=0.3)
    
    print(f'Plot saved as {args.output}')
    print(f'Initial LR: {args.learning_rate:.1e}, Final LR: {args.lr_end:.1e}')
    print(f'Scale type: {args.log_scale}, Notation: {"scientific" if args.notation == "s" else "decimal"}')

def main():
    parser = setup_parser()
    
    # Print help if no arguments provided
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    
    args = parser.parse_args()
    plot_schedules(args)

if __name__ == "__main__":
    main()
