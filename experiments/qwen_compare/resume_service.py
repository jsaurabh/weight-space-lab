"""Spawn-safe entry point for a restored numerical worker."""
from backend import Service


def main():
    service = Service()
    service.maintenance_notice = (
        'Restored your exact saved weights, checkpoints and observations after documented '
        'infrastructure maintenance. Temporary CPU variables were cleared; failed generation '
        'requests were refunded. Remaining budgets are shown in each tool result. '
        'The numerical backend and tools are unchanged. '
        'No peer repair or final test data was provided.'
    )
    service.run()


if __name__ == '__main__':
    main()
