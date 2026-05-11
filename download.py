import cdsapi

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-pressure-levels',
    {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': [
            'fraction_of_cloud_cover',
            'specific_cloud_liquid_water_content',
            'specific_cloud_ice_water_content',
            'specific_rain_water_content',
            'specific_snow_water_content',
            'relative_humidity',
        ],
        'pressure_level': [
            '100', '125', '150', '175', '200', '225', '250',
            '300', '350', '400', '450', '500', '550', '600',
            '650', '700', '750', '775', '800', '825', '850',
            '875', '900', '925', '950', '975', '1000',
        ],
        'year': '2017',
        'month': '08',
        'day': '25',
        'time': '18:00',
        # Bounding box: [North, West, South, East]
        # Gulf Coast crop covering Harvey at landfall
        'area': [35, -105, 15, -80],
        'grid': [0.25, 0.25],
    },
    'harvey_landfall.nc'
)
