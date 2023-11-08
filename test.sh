cd ../MVSFormer

CUDA_VISIBLE_DEVICES=0 python test.py \
	--dataset dtu \
	--batch_size 1 \
	--testpath /home/chli/chLi/Dataset/MVSFormer/DTU/dtu_testing/dtu/ \
	--testlist ../MVSFormer/lists/dtu/test.txt \
	--resume /home/chli/chLi/Model/MVSFormer/MVSFormer/MVSFormer/best.pth \
	--outdir ../mvs-former/output/ \
	--fusibile_exe_path ../mvs-former/mvs_former/Lib/fusibile/build/fusibile \
	--interval_scale 1.06 \
	--num_view 5 \
	--numdepth 192 \
	--max_h 1152 \
	--max_w 1536 \
	--filter_method gipuma \
	--disp_threshold 0.1 \
	--num_consistent 2 \
	--prob_threshold 0.5,0.5,0.5,0.5 \
	--combine_conf \
	--tmps 5.0,5.0,5.0,1.0
