import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec



time_granularities_icews14 =  np.array([2,3,7,15,30,90,365])
time_granularities_yago12k = np.array([1,10,100,1000,10000])
log_time_granularities_yago12k = np.log(time_granularities_yago12k)
dimensions = np.array([20,50,100,200,500,1000])

# icews14 results
mrr_dimensions_icews14 = np.array([0.48079,0.52163,0.56218,0.59046,0.61231,0.61996])
hits1_dimensions_icews14 = np.array([0.3777,0.4299,0.4775,0.5080,0.5319,0.5401])

mrr_timegrans_icews14 = np.array([0.60841,0.59951,0.57444,0.56049,0.53857,0.50614,0.47112])
hits1_timegrans_icews14 = np.array([0.5156, 0.5012, 0.4673, 0.4492, 0.4227, 0.3862, 0.3512])

# yago12k results
mrr_timegrans_yago12k = np.array([0.21812, 0.218596,0.216563,0.210776,0.193725])
hits1_timegrans_yago12k = np.array([0.1573, 0.1588, 0.1559, 0.1507, 0.1378])

mrr_dimensions_yago12k = np.array([0.171582,0.185287,0.1913437,0.200834,0.21023,0.213937])
hits1_dimensions_yago12k = np.array([0.1327, 0.1385, 0.1410, 0.1476, 0.1529, 0.1556])



##### plot #####
# plt.style.use(['science'])

fig = plt.figure(constrained_layout=True)
gs = gridspec.GridSpec(2, 2, figure=fig)
# icews14 time grans
ax0 = fig.add_subplot(gs[0, 0])
ax0.plot(time_granularities_icews14, mrr_timegrans_icews14, label="MRR")
ax0.plot(time_granularities_icews14, hits1_timegrans_icews14, label="Hits@1")
ax0.grid(True)
ax0.legend()
ax0.set_xlabel('time granularities')
plt.title("time granularity ICEWS14")


# yago time grans
ax1 = fig.add_subplot(gs[0, 1])
ax1.set_xscale('log')
ax1.plot(time_granularities_yago12k, mrr_timegrans_yago12k, label="MRR")
ax1.plot(time_granularities_yago12k, hits1_timegrans_yago12k, label="Hits@1")

ax1.grid(True)
ax1.legend()
ax1.set_xlabel('time granularities')
plt.title("time granularity YAGO12k")



# icews14 dimensions
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(dimensions, mrr_dimensions_icews14, label="MRR")
ax3.plot(dimensions, hits1_dimensions_icews14, label="Hits@1")
ax3.grid(True)
ax3.legend()
ax3.set_xlabel('dimensionalities')
plt.title("dimensions ICEWS14")


# yago dimensions
ax2 = fig.add_subplot(gs[1, 1])
ax2.plot(dimensions, mrr_dimensions_yago12k, label="MRR")
ax2.plot(dimensions, hits1_dimensions_yago12k, label="Hits@1")
ax2.grid(True)
ax2.legend()
ax2.set_xlabel('dimensionalities')
plt.title("dimensions YAGO12k")

plt.show()

