import predict_interface

"""
How to use the predict_interface -interface
"""

val = -1


def train_and_obtain_number(img_source):
	global val
	ret_vals = predict_interface.pred_from_img(img_source, False)

	print("----------")

	for i in range(len(ret_vals)):
		print("Prediction: " + str(ret_vals[i].get_prediction()) + ", Probability: " + str(ret_vals[i].get_probability()))
		val = ret_vals[i].get_prediction()
		return val
	else:
		return -1

