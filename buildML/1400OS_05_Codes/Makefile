CHART_DIR = ../charts
TARGET_DIR = /media/sf_P/Dropbox/pymlbook/pmle/ch05

copy: 
	cp $(CHART_DIR)/feat_hist_LinkCount.png $(TARGET_DIR)/1400_05_01.png
	cp $(CHART_DIR)/feat_hist_two.png $(TARGET_DIR)/1400_05_02.png
	cp $(CHART_DIR)/feat_hist_four.png $(TARGET_DIR)/1400_05_03.png
	
	cp $(CHART_DIR)/bv_5NN.png $(TARGET_DIR)/1400_05_04.png
	#cp $(CHART_DIR)/bv_5NN.png $(TARGET_DIR)/1400_05_05.png
	cp $(CHART_DIR)/kcomplexity.png $(TARGET_DIR)/1400_05_06.png 
	
	cp log_reg_example_data.png $(TARGET_DIR)/1400_05_07.png
	cp log_reg_log_odds.png $(TARGET_DIR)/1400_05_08.png
	cp log_reg_example_fitted.png $(TARGET_DIR)/1400_05_09.png
	
	cp $(CHART_DIR)/bv_LogReg_C=0.10.png $(TARGET_DIR)/1400_05_10.png
	convert $(CHART_DIR)/pr_LogReg_C=0.10_poor.png $(CHART_DIR)/pr_LogReg_C=0.10_good.png +append $(TARGET_DIR)/1400_05_11.png
	cp $(CHART_DIR)/feat_imp_LogReg_C=0.10.png $(TARGET_DIR)/1400_05_13.png
	
	cp *.py $(TARGET_DIR)/code
	cp Makefile $(TARGET_DIR)/code

01:
	python 01_start.py

02:
	python 02_tuning.py

03:
	python 03_clean.py

04:
	python 04_sent.py

log_probs.png:
	python utils.py

rocs_ceps.png: 
	convert $(CHART_DIR)/roc_Log_Reg_CEPS_classical.png $(CHART_DIR)/roc_Log_Reg_CEPS_jazz.png +append row1.png
	convert $(CHART_DIR)/roc_Log_Reg_CEPS_country.png $(CHART_DIR)/roc_Log_Reg_CEPS_pop.png +append row2.png
	convert $(CHART_DIR)/roc_Log_Reg_CEPS_rock.png $(CHART_DIR)/roc_Log_Reg_CEPS_metal.png +append row3.png
	convert row1.png row2.png row3.png -append $(CHART_DIR)/rocs_ceps.png
	
roc_pr.png: fft
	convert $(CHART_DIR)/pr_Log_Reg_FFT_country.png $(CHART_DIR)/roc_Log_Reg_FFT_country.png +append roc_pr.png

sox sine_a.wav sine_b.wav sine_mix.wav:
	sox --null -r 22050 sine_a.wav synth 0.2 sine 400
	sox --null -r 22050 sine_b.wav synth 0.2 sine 3000
	sox --combine mix --volume 1 sine_b.wav --volume 0.5 sine_a.wav sine_mix.wav

fft_demo: sine_a.wav sine_b.wav sine_mix.wav
	python fft.py 
	convert sine_a_wav_fft.png sine_b_wav_fft.png sine_mix_wav_fft.png -append fft_demo.png
	
	python fft.py /media/sf_P/pymlbook-data/09-genre-class/genres/jazz/jazz.00012.wav
	mv jazz.00012_wav_fft.png fft_example.png
	

