from graph import gen_export_db
from kernels import *
db2, path = gen_export_db(3,3,10,5,0.05)
test(db2)
