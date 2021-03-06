FROM ubuntu:14.04

ENV TIMEZONE=Asia/Shanghai
ENV FLREC "/flrec"

ENV FL_ENV "develop"  # production|develop|localhost
ENV GEVENT_POOL "4"

ENV ACCESS_KEY_ID ""
ENV ACCESS_KEY_SECRET ""
ENV ODPS_ENDPOINT "http://service.odps.aliyun.com/api"

ENV DAY7_DATA "${FLREC}/data/7day_data"

ADD . ${FLREC}
#COPY aliyun.sources.list /etc/apt/sources.list

WORKDIR ${FLREC}

RUN set -x; \
        apt-get update \
     && apt-get -y install \
            curl \
            tzdata \
            python-dev \
            python-pip \
            python-setuptools \
            libev-dev \
            libblas-dev \
            liblapack-dev \
            libatlas-base-dev \
            gfortran \
     && rm -rf /usr/share/doc /usr/share/man /var/lib/apt/lists/* \
     && pip install --timeout=100 -r requirements.txt --upgrade; \
        sed -i 's#^UTC=.*#UTC=no#g' /etc/default/rcS \
     && python setup.py install \
     && echo "${TIMEZONE}" > /etc/timezone \
     && cp /usr/share/zoneinfo/${TIMEZONE} /etc/localtime

WORKDIR ${FLREC}

VOLUME "/flrec/data/"

CMD ["python", "examples/flrec.py"]

