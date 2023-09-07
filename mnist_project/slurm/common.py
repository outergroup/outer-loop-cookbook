import enum

# include for compatibility with old pkl files...
Phase = enum.Enum("Phase", ["BO", "TRIAL", "DONE"])
