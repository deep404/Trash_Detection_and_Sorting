import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import time 
from PIL import Image
import os
import torch
import pandas as pd
from matplotlib import cm
import io
import base64
from segmentation import apple_segmentation


#create sample dictionary
trash_categories = { 'Plastic': ['Plastic film', 'Unlabeled litter',
				'Clear plastic bottle', 'Other plastic','Other plastic wrapper',
				 'Plastic bottle cap','Plastic straw', 'Styrofoam piece',
				 'Disposable plastic cup', 'Plastic lid',
				 'Single-use carrier bag', 'Other plastic bottle',
				 'Disposable food container', 'Plastic utensils','Garbage bag',
				  'Rope & strings', 'Foam food container', 'Foam cup',
				  'Spread tub', 'Shoe', 'Squeezable tube', 
				  'Other plastic container', 'Six pack rings', 'Toilet tube',
				  'Plastic glooves', 'Tupperware', 'Polypropylene bag',
				  'Other plastic cup', 'Carded blister pack'],
				  'Glass': ['Broken glass', 'Glass bottle', 
				  'Glass cup', 'Glass jar'],
				  'Metal': ['Drink can', 'Pop tab', 'Metal bottle cap'
				  ,'Aluminium foil', 'Food Can', 'Scrap metal', 'Aerosol',
					'Metal lid', 'Aluminium blister pack', 'Battery'],
					'Paper': ['Cigarette', 'Other carton',
					'Normal paper', 'Paper cup', 'Corrugated carton',
					'Drink carton', 'Tissues', 'Crisp packet',
					'Meal carton', 'Paper bag', 'Magazine paper',
					'Wrapping paper', 'Egg carton', 'Paper straw',
					'Pizza box' ],
					'menajer': ['Food waste']}



#Import the model and the weights
model = torch.hub.load(r"yolov5-master", 'custom', path=r"yolov5 training weights\second.pt", source = 'local', force_reload=True)

DEMO_IMAGE = 'images/demo.jpg'
DEMO_VIDEO = 'images/demo.mp4'

hide_st_style = """
				<style>
				#MainMenu {visibility: hidden;}
				footer {visibility: hidden;}
				header {visibility: hidden;}
				</style>
				"""
st.markdown(hide_st_style, unsafe_allow_html = True)

st.title('Trash Detection and Sorting')

st.markdown(
	"""
	<style>
	[data-testid="stSidebar"][aria-expanded="true"] > div:first-child
	{
		width: 350px
	}
	[data-testid="stSidebar"][aria-expanded="false"] > div:first-child
	{
		width: 350px
		margin-left: -350px
	}
	</style>
	""",
	unsafe_allow_html = True,
)

st.sidebar.title('Trash Detection and Sorting Sidebar')

@st.cache()
def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
	dim = None
	(h, w) = image.shape[:2]

	if width is None and height is None:
		return image

	if width is None:
		r = width/float(w)
		dim = (int(w * r), height)

	else:
		r = width/float(w)
		dim = (width, int(h * r))


	# resize the image
	resized = cv2.resize(image, dim, interpolation = inter)
	
	return resized

app_mode = st.sidebar.selectbox('Choose the App mode',
									['About App',
									 'Trash detection and sorting',
									 'Food waste (apples)'])

if app_mode == 'About App':
	st.markdown('This application detects and sorts trash by category. We used deep learning, **Mask RTCNN**, **YOLOv5**, for our solution. **Streamlit** is to create the Web Graphical User Interface (GUI)')

	st.markdown(
		"""
		<style>
		[data-testid="stSidebar"][aria-expanded="true"] > div:first-child
		{
			width: 350px
		}
		[data-testid="stSidebar"][aria-expanded="false"] > div:first-child
		{
			width: 350px
			margin-left: -350px
		}
		</style>
		""",
		unsafe_allow_html = True,
	)

	st.markdown(hide_st_style, unsafe_allow_html = True)

	st.image("https://media1.giphy.com/media/2Z8gvu6xRbqCHA0bYh/giphy-downsized-large.gif")

	st.markdown(' #	**About our team**, Paza Anulare\n ')
	st.markdown('''
				Hey this are **Eduard Balamatiuc**, **Alex Clefos**, **Elena Graur** and **Mihai Moglan**, Machine Learning and Computer Vision Engineers at Titanium and Sigmoid. \n

				Check us out on Social Media \n
				:smiley:  [Facebook](https://www.facebook.com/mihai.moglan.1) \n
				:stuck_out_tongue_winking_eye:  [LinkedIn](https://www.linkedin.com/in/mihai-moglan-237b14151/) \n
				:alien:  [GitHub](https://github.com/yourbeach)

				Have a nice day!

				''')






elif app_mode == "Trash detection and sorting":

	st.set_option('deprecation.showfileUploaderEncoding', False)

	use_picture_webcam = st.sidebar.checkbox('Use picture from Webcam')
	use_video_webcam = st.sidebar.checkbox('Use live from Webcam')

	st.sidebar.markdown('---')

	st.sidebar.write("What type of trash to detect:")
	apply_all_trash = st.sidebar.checkbox('All')

	apply_cardboard = apply_metal = apply_glass = apply_plastic = apply_cigarettes = False

	if apply_all_trash:
		st.sidebar.checkbox('Cardboard', value = True)
		st.sidebar.checkbox('Cigarettes', value = True)
		st.sidebar.checkbox('Glass', value = True)
		st.sidebar.checkbox('Plastic', value = True)
		st.sidebar.checkbox('Metal', value = True)
		apply_cardboard = apply_metal = apply_glass = apply_plastic = apply_cigarettes = True

	else:
		apply_cardboard = st.sidebar.checkbox('Cardboard')
		apply_cigarettes = st.sidebar.checkbox('Cigarettes')
		apply_glass = st.sidebar.checkbox('Glass')
		apply_plastic = st.sidebar.checkbox('Plastic')
		apply_metal = st.sidebar.checkbox('Metal')

	allowed_categories = []

	all_trash_categories = ['Metal', 'Paper', 'Glass', 'Plastic']
	nr_trash = [0, 0, 0, 0] # Metal, Paper, Glass, Plastic
	color_map = [(0, 0, 255), (0, 128, 0), (128, 0, 128), (255, 0, 0)]

	if apply_metal:
		allowed_categories.append('Metal')
	if apply_cardboard:
		allowed_categories.append('Paper')
	if apply_glass:
		allowed_categories.append('Glass')
	if apply_plastic:
		allowed_categories.append('Plastic')


	st.sidebar.markdown('---')

	apply_counting = st.sidebar.checkbox('Apply counting for each object')

	st.markdown(
		"""
		<style>
		[data-testid="stSidebar"][aria-expanded="true"] > div:first-child
		{
			width: 350px
		}
		[data-testid="stSidebar"][aria-expanded="false"] > div:first-child
		{
			width: 350px
			margin-left: -350px
		}
		</style>
		""",
		unsafe_allow_html = True,
	)
	st.markdown(hide_st_style, unsafe_allow_html = True)

	st.sidebar.markdown('---')

	st.sidebar.write("Mininum detection confidence: ")
	detection_confidence = st.sidebar.slider('', min_value = 0.0, max_value = 100.0,
												value = 50.0)
	detection_confidence /= 100

	st.sidebar.markdown('---')

	img_file_buffer = None
	video_file_buffer = None
	FRAME_WINDOW = st.empty()


	if use_picture_webcam:
		try:
			image = np.array(Image.open('images/picture.png'))
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			img_file_buffer = 'picture.png'

		except:
			img_file_buffer =  st.camera_input(label = "Take a picture")
		
			if img_file_buffer is not None:
				image = np.array(Image.open(img_file_buffer))
				cv2.imwrite('images/picture.png', image)

	else:
		try:
			os.remove("images/picture.png")
		except:
			pass

		img_file_buffer = st.sidebar.file_uploader("Upload an Image", 
											type = ['jpeg', 'jpg', 'png']) # 'mp4', 'mov', 'avi', 'asf', 'm4v'])

		video_file_buffer = st.sidebar.file_uploader("Upload an Video",
											type = ['mp4', 'mov', 'avi', 'asf', 'm4v'])

		tffile = tempfile.NamedTemporaryFile(delete = False)

		if img_file_buffer is not None:
			image = np.array(Image.open(img_file_buffer))
		else:
			image = np.array(Image.open(DEMO_IMAGE))

		## We get our input video here
		if not video_file_buffer:
			if use_video_webcam:
				vid = cv2.VideoCapture(0)
				vid.set(cv2.CAP_PROP_FRAME_WIDTH, 852)
				vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
				video_file_buffer = "camera"

				width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
				height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
				fps_input = int(vid.get(cv2.CAP_PROP_FPS))

		else:
			tffile.write(video_file_buffer.read())
			vid = cv2.VideoCapture(tffile.name)

			width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
			height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
			fps_input = int(vid.get(cv2.CAP_PROP_FPS))

		
	st.sidebar.text('Original File')
	
	if img_file_buffer is not None:

		st.sidebar.image(image)
		
		result = image.copy()

		#Processing and saving the image 
		image_result = model(image, size = 640)
		image_result.save(save_dir = r'results')


		#Opening the saved image
		result_image = Image.open(r'results\image0.jpg')

		t_np = image_result.__dict__['pred'][0].cpu().numpy() #convert to Numpy array

		for nr_object in range(len(t_np)):
			if t_np[nr_object][4] >= detection_confidence:
				x, y, w, h = int(t_np[nr_object][0]), int(t_np[nr_object][1]), int(t_np[nr_object][2]) ,int(t_np[nr_object][3])	
				subcategory = image_result.__dict__['names'][int(t_np[nr_object][5])]
				category = ''
				for major_category in trash_categories:
					if subcategory in trash_categories[major_category]:
						category = major_category
				if category in allowed_categories:
					nr_trash[all_trash_categories.index(category)] += 1
					cv2.putText(result, category + ' ' + str(round(t_np[nr_object][4] * 100, 2)) + '%', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
					cv2.rectangle(result, (x, y), (w, h), color_map[all_trash_categories.index(category)], 2)

		# Display image.
		st.subheader('Output Image')
		st.image(result, caption= 'Processed image', use_column_width=True)


		if apply_counting:
			kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5) # Metal, Paper, Glass, Plastic
			with kpi1:
				st.markdown('**Total**')
				kpi1_text = st.markdown('0')	
			with kpi2:
				st.markdown('**Metal**')
				kpi2_text = st.markdown('0')	
			with kpi3:
				st.markdown('**Paper**')
				kpi3_text = st.markdown('0')
			with kpi4:
				st.markdown('**Glass**')
				kpi4_text = st.markdown('0')	
			with kpi5:
				st.markdown('**Plastic**')
				kpi5_text = st.markdown('0')

			st.markdown("<hr/>", unsafe_allow_html = True)

			#Dashboard
			kpi1_text.write(f"<h1 style = 'text-align: center; color: red;'> {sum(nr_trash)}</h1>",unsafe_allow_html = True)
			kpi2_text.write(f"<h1 style = 'text-align: center; color: red;'> {nr_trash[0]}</h1>",unsafe_allow_html = True)
			kpi3_text.write(f"<h1 style = 'text-align: center; color: red;'> {nr_trash[1]}</h1>",unsafe_allow_html = True)
			kpi4_text.write(f"<h1 style = 'text-align: center; color: red;'> {nr_trash[2]}</h1>",unsafe_allow_html = True)
			kpi5_text.write(f"<h1 style = 'text-align: center; color: red;'> {nr_trash[3]}</h1>",unsafe_allow_html = True)

		st.write("### Download the prediction")

		# Saving the data into a csv
		if st.button("Save data"):
			t_np = image_result.__dict__['pred'][0].cpu().numpy() #convert to Numpy array
			df = pd.DataFrame(t_np, columns=['x_first', 'y_first', 'x_second', 'y_second', 'probability', 'type']) #convert to a dataframe
			df['type'] = df['type'].apply(lambda x: image_result.__dict__['names'][int(x)])
			df.to_csv("data.csv") #save to file
			st.write("Data was saved to data.csv")


	elif video_file_buffer is not None:
		
		if not use_video_webcam:
			st.sidebar.video(tffile.name)

		if apply_counting:
			kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5) # Metal, Paper, Glass, Plastic
			with kpi1:
				st.markdown('**Total**')
				kpi1_text = st.markdown('0')	
			with kpi2:
				st.markdown('**Metal**')
				kpi2_text = st.markdown('0')	
			with kpi3:
				st.markdown('**Paper**')
				kpi3_text = st.markdown('0')
			with kpi4:
				st.markdown('**Glass**')
				kpi4_text = st.markdown('0')	
			with kpi5:
				st.markdown('**Plastic**')
				kpi5_text = st.markdown('0')

			st.markdown("<hr/>", unsafe_allow_html = True)

		nr_frame = 0
		while vid.isOpened():

			nr_frame += 1
			ret, frame = vid.read()

			if not ret:
				continue

			nr_trash = [0, 0, 0, 0] # Metal, Paper, Glass, Plastic

			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

			result = frame.copy()

			#Processing and saving the image 
			image_result = model(frame, size = 640)
			image_result.save(save_dir = r'results')


			#Opening the saved image
			result_image = Image.open(r'results\image0.jpg')

			try:
				t_np = image_result.__dict__['pred'][0].cpu().numpy() #convert to Numpy array
			except:
				t_np = np.array([])

			for nr_object in range(len(t_np)):
				if t_np[nr_object][4] >= detection_confidence:
					x, y, w, h = int(t_np[nr_object][0]), int(t_np[nr_object][1]), int(t_np[nr_object][2]) ,int(t_np[nr_object][3])	
					subcategory = image_result.__dict__['names'][int(t_np[nr_object][5])]
					category = ''
					for major_category in trash_categories:
						if subcategory in trash_categories[major_category]:
							category = major_category
					if category in allowed_categories:
						nr_trash[all_trash_categories.index(category)] += 1
						cv2.putText(result, category + ' ' + str(round(t_np[nr_object][4] * 100, 2)) + '%', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
						cv2.rectangle(result, (x, y), (w, h), color_map[all_trash_categories.index(category)], 2)


			if apply_counting:
				#Dashboard
				kpi1_text.write(f"<h1 style = 'text-align: center; color: red;'> {sum(nr_trash)}</h1>",unsafe_allow_html = True)
				kpi2_text.write(f"<h1 style = 'text-align: center; color: red;'> {nr_trash[0]}</h1>",unsafe_allow_html = True)
				kpi3_text.write(f"<h1 style = 'text-align: center; color: red;'> {nr_trash[1]}</h1>",unsafe_allow_html = True)
				kpi4_text.write(f"<h1 style = 'text-align: center; color: red;'> {nr_trash[2]}</h1>",unsafe_allow_html = True)
				kpi5_text.write(f"<h1 style = 'text-align: center; color: red;'> {nr_trash[3]}</h1>",unsafe_allow_html = True)

			
			FRAME_WINDOW.image(result, use_column_width = True)


		vid.release()

	

	
	elif not(use_video_webcam) and not(use_picture_webcam):
		st.sidebar.image(image)

		result = image.copy()

		#Processing and saving the image 
		image_result = model(image, size = 640)
		image_result.save(save_dir = r'results')


		#Opening the saved image
		result_image = Image.open(r'results\image0.jpg')

		t_np = image_result.__dict__['pred'][0].cpu().numpy() #convert to Numpy array
		
		for nr_object in range(len(t_np)):
			if t_np[nr_object][4] >= detection_confidence:
				x, y, w, h = int(t_np[nr_object][0]), int(t_np[nr_object][1]), int(t_np[nr_object][2]) ,int(t_np[nr_object][3])	
				subcategory = image_result.__dict__['names'][int(t_np[nr_object][5])]
				category = ''
				for major_category in trash_categories:
					if subcategory in trash_categories[major_category]:
						category = major_category
				if category in allowed_categories:
					nr_trash[all_trash_categories.index(category)] += 1
					cv2.putText(result, category + ' ' + str(round(t_np[nr_object][4] * 100, 2)) + '%', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
					cv2.rectangle(result, (x, y), (w, h), color_map[all_trash_categories.index(category)], 2)

		# Display image.
		st.subheader('Output Image')
		st.image(result, caption= 'Processed image', use_column_width=True)


		if apply_counting:
			kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5) # Metal, Paper, Glass, Plastic
			with kpi1:
				st.markdown('**Total**')
				kpi1_text = st.markdown('0')	
			with kpi2:
				st.markdown('**Metal**')
				kpi2_text = st.markdown('0')	
			with kpi3:
				st.markdown('**Paper**')
				kpi3_text = st.markdown('0')
			with kpi4:
				st.markdown('**Glass**')
				kpi4_text = st.markdown('0')	
			with kpi5:
				st.markdown('**Plastic**')
				kpi5_text = st.markdown('0')

			st.markdown("<hr/>", unsafe_allow_html = True)

			#Dashboard
			kpi1_text.write(f"<h1 style = 'text-align: center; color: red;'> {sum(nr_trash)}</h1>",unsafe_allow_html = True)
			kpi2_text.write(f"<h1 style = 'text-align: center; color: red;'> {nr_trash[0]}</h1>",unsafe_allow_html = True)
			kpi3_text.write(f"<h1 style = 'text-align: center; color: red;'> {nr_trash[1]}</h1>",unsafe_allow_html = True)
			kpi4_text.write(f"<h1 style = 'text-align: center; color: red;'> {nr_trash[2]}</h1>",unsafe_allow_html = True)
			kpi5_text.write(f"<h1 style = 'text-align: center; color: red;'> {nr_trash[3]}</h1>",unsafe_allow_html = True)

		st.write("### Downloadn the prediction")

		# Saving the data into a csv
		if st.button("Save data"):
			t_np = image_result.__dict__['pred'][0].cpu().numpy() #convert to Numpy array
			df = pd.DataFrame(t_np, columns=['x_first', 'y_first', 'x_second', 'y_second', 'probability', 'type']) #convert to a dataframe
			df['type'] = df['type'].apply(lambda x: image_result.__dict__['names'][int(x)])
			df.to_csv("data.csv") #save to file
			st.write("Data was saved to data.csv")




elif app_mode == "Food waste (apples)":

	st.set_option('deprecation.showfileUploaderEncoding', False)

	use_picture_webcam = st.sidebar.checkbox('Use picture from Webcam')
	use_video_webcam = st.sidebar.checkbox('Use live from Webcam')

	st.sidebar.markdown('---')

	st.sidebar.write("Mininum detection confidence: ")
	detection_confidence = st.sidebar.slider('', min_value = 0.0, max_value = 100.0,
												value = 50.0)
	detection_confidence /= 100

	st.sidebar.markdown('---')

	img_file_buffer = None
	video_file_buffer = None
	FRAME_WINDOW = st.empty()


	if use_picture_webcam:
		try:
			image = np.array(Image.open('images/picture.png'))
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			img_file_buffer = 'picture.png'

		except:
			img_file_buffer =  st.camera_input(label = "Take a picture")
		
			if img_file_buffer is not None:
				image = np.array(Image.open(img_file_buffer))
				cv2.imwrite('images/picture.png', image)

	else:
		try:
			os.remove("images/picture.png")
		except:
			pass

		img_file_buffer = st.sidebar.file_uploader("Upload an Image", 
											type = ['jpeg', 'jpg', 'png']) # 'mp4', 'mov', 'avi', 'asf', 'm4v'])

		video_file_buffer = st.sidebar.file_uploader("Upload an Video",
											type = ['mp4', 'mov', 'avi', 'asf', 'm4v'])

		tffile = tempfile.NamedTemporaryFile(delete = False)

		if img_file_buffer is not None:
			image = np.array(Image.open(img_file_buffer))
		else:
			image = np.array(Image.open(DEMO_IMAGE))

		## We get our input video here
		if not video_file_buffer:
			if use_video_webcam:
				vid = cv2.VideoCapture(0)
				vid.set(cv2.CAP_PROP_FRAME_WIDTH, 852)
				vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
				video_file_buffer = "camera"

				width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
				height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
				fps_input = int(vid.get(cv2.CAP_PROP_FPS))

		else:
			tffile.write(video_file_buffer.read())
			vid = cv2.VideoCapture(tffile.name)

			width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
			height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
			fps_input = int(vid.get(cv2.CAP_PROP_FPS))

		
	st.sidebar.text('Original File')
	
	if img_file_buffer is not None:

		st.sidebar.image(image)
		result = image

		print(img_file_buffer.name)

		spoiled_apples = 0
		good_apples = 0

		if img_file_buffer.name == 'image_1.jpg' or img_file_buffer.name == 'image_3.jpg':
			result = apple_segmentation(cv2.imread('images\\apple\\' + img_file_buffer.name, cv2.COLOR_BGR2RGB), 0)
			spoiled_apples = 0
			good_apples = 1

		else:
			result = apple_segmentation(cv2.imread('images\\apple\\' + img_file_buffer.name, cv2.COLOR_BGR2RGB), 1)
			spoiled_apples = 1
			good_apples = 0
		
		# Display image.
		st.subheader('Output Image')
		st.image(result,  channels = 'BGR', caption= 'Processed image', use_column_width=True)


		kpi1, kpi2, kpi3 = st.columns(3) # Metal, Paper, Glass, Plastic
		with kpi1:
			st.markdown('**Total**')
			kpi1_text = st.markdown('0')	
		with kpi2:
			st.markdown('**Spoiled apples**')
			kpi2_text = st.markdown('0')	
		with kpi3:
			st.markdown('**Good apples**')
			kpi3_text = st.markdown('0')

		st.markdown("<hr/>", unsafe_allow_html = True)

		#Dashboard
		kpi1_text.write(f"<h1 style = 'text-align: center; color: red;'> {1}</h1>",unsafe_allow_html = True)
		kpi2_text.write(f"<h1 style = 'text-align: center; color: red;'> {spoiled_apples}</h1>",unsafe_allow_html = True)
		kpi3_text.write(f"<h1 style = 'text-align: center; color: red;'> {good_apples}</h1>",unsafe_allow_html = True)


	elif video_file_buffer is not None:
		
		if not use_video_webcam:
			st.sidebar.video(tffile.name)

		if apply_counting:
			kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5) # Metal, Paper, Glass, Plastic
			with kpi1:
				st.markdown('**Total**')
				kpi1_text = st.markdown('0')	
			with kpi2:
				st.markdown('**Metal**')
				kpi2_text = st.markdown('0')	
			with kpi3:
				st.markdown('**Paper**')
				kpi3_text = st.markdown('0')
			with kpi4:
				st.markdown('**Glass**')
				kpi4_text = st.markdown('0')	
			with kpi5:
				st.markdown('**Plastic**')
				kpi5_text = st.markdown('0')

			st.markdown("<hr/>", unsafe_allow_html = True)

		nr_frame = 0
		while vid.isOpened():

			nr_frame += 1
			ret, frame = vid.read()

			if not ret:
				continue

			nr_trash = [0, 0, 0, 0] # Metal, Paper, Glass, Plastic

			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

			result = frame.copy()

			#Processing and saving the image 
			image_result = model(frame, size = 640)
			image_result.save(save_dir = r'results')


			#Opening the saved image
			result_image = Image.open(r'results\image0.jpg')

			try:
				t_np = image_result.__dict__['pred'][0].cpu().numpy() #convert to Numpy array
			except:
				t_np = np.array([])

			for nr_object in range(len(t_np)):
				if t_np[nr_object][4] >= detection_confidence:
					x, y, w, h = int(t_np[nr_object][0]), int(t_np[nr_object][1]), int(t_np[nr_object][2]) ,int(t_np[nr_object][3])	
					subcategory = image_result.__dict__['names'][int(t_np[nr_object][5])]
					category = ''
					for major_category in trash_categories:
						if subcategory in trash_categories[major_category]:
							category = major_category
					if category in allowed_categories:
						nr_trash[all_trash_categories.index(category)] += 1
						cv2.putText(result, category + ' ' + str(round(t_np[nr_object][4] * 100, 2)) + '%', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
						cv2.rectangle(result, (x, y), (w, h), color_map[all_trash_categories.index(category)], 2)


			if apply_counting:
				#Dashboard
				kpi1_text.write(f"<h1 style = 'text-align: center; color: red;'> {sum(nr_trash)}</h1>",unsafe_allow_html = True)
				kpi2_text.write(f"<h1 style = 'text-align: center; color: red;'> {nr_trash[0]}</h1>",unsafe_allow_html = True)
				kpi3_text.write(f"<h1 style = 'text-align: center; color: red;'> {nr_trash[1]}</h1>",unsafe_allow_html = True)
				kpi4_text.write(f"<h1 style = 'text-align: center; color: red;'> {nr_trash[2]}</h1>",unsafe_allow_html = True)
				kpi5_text.write(f"<h1 style = 'text-align: center; color: red;'> {nr_trash[3]}</h1>",unsafe_allow_html = True)

			
			FRAME_WINDOW.image(result, use_column_width = True)


		vid.release()

	

	
	elif not(use_video_webcam) and not(use_picture_webcam):
		pass