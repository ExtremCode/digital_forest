from tifffile import imread, imwrite
import geopandas as gpd
from shapely.geometry import Point, Polygon
from geocube.api.core import make_geocube
import math
import numpy as np
from os import listdir

# CLASSES = sorted(['ABBA', 'ACPE', 'ACRU', 'ACSA', 'Acer', 'BEAL', 
#                   'BEPA', 'BEPO', 'Betula', 'Conifere', 'FAGR',
#                     'FRNI', 'Feuillus', 'LALA', 'Mort', 'PIGL',
#                       'PIMA', 'PIRU', 'PIST', 'POBA', 'POGR',
#                         'POTR', 'PRPE', 'Picea', 'Populus',
#                           'QURU', 'THOC', 'TSCA', 'OSVI'])

labels = sorted(['ABBA', 'ACPE', 'ACRU', 'ACSA', 'BEAL', 
                  'BEPA', 'FAGR', 'LALA', 'Mort', 'PIST', 
                  'Picea', 'Populus', 'THOC', 'TSCA'])


def get_label(lab: str) -> int:
    global labels
    if lab not in labels:
        if lab == 'Acer' or lab == 'PRPE':
            lab = 'ACPE'
        
        elif lab == 'BEPO' or lab == 'Betula':
            lab = 'BEPA'
        
        elif lab == 'Conifere' or lab == 'PIGL' or lab == 'PIMA' or lab == 'PIRU':
            lab = 'Picea'

        elif lab == 'FRNI' or lab == 'POBA' or lab == 'POGR' or lab == 'POTR':
            lab = 'Populus'
        
        elif lab == 'Feuillus' or lab == 'OSVI':
            lab = 'BEAL'

        elif lab == 'QURU':
            lab = 'ACRU'

        else:
            lab = 'TSCA'
    
    return labels.index(lab) + 1


# date = ['09-28']
def cut_and_rast_masks(dates: list[int], num_zones: int):
    for dt in dates:
        for zone in range(1, num_zones + 1):
            pol8 = gpd.GeoDataFrame.from_file(
                f"C:/Users/Лев/Downloads/quebec_trees_dataset_2021-{dt}/quebec_trees_dataset_2021-{dt}/Z{zone}_polygons.gpkg")
            pol8['Label'].astype(str, copy=False)
            pol8['Label'] = pol8['Label'].apply(get_label)

            file = gpd.read_file(f"Grid/grid-z{zone}-{dt}.gpkg")

            for i in range(file.shape[0]):
                seria = file.iloc[i]
                row_index = seria.row_index
                col_index = seria.col_index
                pol1 = gpd.GeoDataFrame(geometry=gpd.GeoSeries(seria.geometry,
                        crs='EPSG:32618')) # first go in each rows then each col

                bbox = pol1.total_bounds

                p1 = Point(bbox[0], bbox[3])
                p2 = Point(bbox[2], bbox[3])
                p3 = Point(bbox[2], bbox[1])
                p4 = Point(bbox[0], bbox[1])

                np1 = (p1.coords.xy[0][0], p1.coords.xy[1][0])
                np2 = (p2.coords.xy[0][0], p2.coords.xy[1][0])
                np3 = (p3.coords.xy[0][0], p3.coords.xy[1][0])
                np4 = (p4.coords.xy[0][0], p4.coords.xy[1][0])

                bb_polygon = Polygon([np1, np2, np3, np4])

                df2 = gpd.GeoDataFrame(gpd.GeoSeries(bb_polygon), 
                                       columns=['geometry'], crs='EPSG:32618')

                intersections2 = gpd.overlay(df2, pol8, how='intersection')
                if not intersections2.empty:
                    difference = df2.overlay(intersections2, how='difference', 
                                             keep_geom_type=False)
                    intersections2 = intersections2.overlay(difference, how='union', 
                                                            keep_geom_type=False)
                else:
                    intersections2 = df2
                    intersections2['Label'] = 0    

                intersections2.fillna(value={'OBJECTID': 0, 'Label': 0}, inplace=True)
                intersections2_raster = make_geocube(
                        vector_data=intersections2,
                        measurements=["Label"],
                        resolution=(-0.019, 0.019),
                        fill = 0
                )
                where = ''
                if i % 5 != 0 and (i % 6 == 0):
                    where = 'Valid'
                elif i % 5 == 0 and i > 0:
                    where = 'Test'
                else:
                    where = 'Train'
                # Save raster
                intersections2_raster.rio.to_raster(
                    f"{where}/mask/mask-z{zone}-{dt}-{col_index}-{row_index}.tif"
                )


def cut_images(dates: list[str], number_of_zones: int):
    for date in dates:
        for zone in range(1, number_of_zones + 1):
            image = imread(f"Source_tif/crop-z{zone}-{date}.tif")
            rows = math.ceil(image.shape[0]/264)
            cols = math.ceil(image.shape[1]/264)
            glob_index = -1
            # outer loop for width (columns)
            # inner loop for height (rows)     
            for j in range(cols):
                for i in range(rows):
                    glob_index += 1
                    where = 'Train'
                    if glob_index % 5 != 0 and glob_index % 6 == 0:
                        where = 'Valid'
                    elif glob_index % 5 == 0 and glob_index > 0:
                        where = 'Test'
                    
                    if i == rows - 1 or j == cols - 1:
                        imwrite(f"{where}/image/crop-z{zone}-{date}-{j}-{i}.tif",\
                                np.zeros((264, 264, 3)), photometric="rgb")
                    else:
                        imwrite(f"{where}/image/crop-z{zone}-{date}-{j}-{i}.tif", \
                            image[i*264:i*264 + 264, j*264:j*264 + 264, :3],
                            photometric="rgb")


def create_proba_file():
    for folder in ["Train", "Test"]:
        for mask in listdir(path=f'{folder}/mask/'):
            img_array = imread(f"{folder}/mask/"+mask)
            unique, counts = np.unique(img_array, return_counts=True)
            length = img_array.size
            dict_classes = dict(zip(unique, counts))
            cnt_classes = [0]*3
            for key, value in dict_classes.items():
                if key == 0 or key == 9:
                    cnt_classes[0] += value
                elif key == 1 or key == 8 or key == 11 \
                    or key == 10 or key == 14:
                    cnt_classes[1] += value
                else:
                    cnt_classes[2] += value
            with open(f"{folder}/proba/proba{mask[
                mask.find("mask")+4:mask.find(".tif")]}.txt", 'x') as file:
                file.write(' '.join([str(el/length) for el in cnt_classes]))


if __name__=="__main__":
    dates = ['09-02', '06-17', '08-18', '07-21', '05-28', '10-07', '09-28']
