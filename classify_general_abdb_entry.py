"""
Classify the CDR conformation of an input abdb entry into canonical classes 
using classifiers trained on unbound CDR conformations.
"""
from cdrclass.app import cli, main 

# ==================== Main ====================
if __name__ == "__main__":
    args = cli()
    main(args=args)