def transfer(src_img_path, dst_img_path, target_img_path):
    from color_transfer import color_transfer, image_stats
    import cv2, uuid

    source = cv2.imread(src_img_path)
    target = cv2.imread(dst_img_path)
    transfer = color_transfer(source, target)
    cv2.imwrite(target_img_path, transfer)
