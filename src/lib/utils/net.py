import os

from typing import List, Optional
from pathlib import Path
import gswrap
import datetime
import time
import logging
from dateutil.tz import tzlocal

from google.api_core.exceptions import ServerError

from requests.exceptions import (
    ReadTimeout,
    ConnectTimeout,
    ConnectionError,
    TooManyRedirects,
    InvalidURL,
    ChunkedEncodingError,
    ContentDecodingError,
    SSLError,
    ProxyError,
    HTTPError,
    MissingSchema,
    RetryError,
)

client = gswrap.Client()

logger = logging.getLogger("gcs-sync")

def get_modified_date_or_none_gcs(url: str) -> Optional[datetime.datetime]:
    """
    When possible return the modified date

    return: datetime object or None
    """
    stat = client.stat(url=url)
    if stat is None:
        return None
    result = stat.creation_time
    return result


def get_modified_date_or_none_fs(path: Path) -> Optional[datetime.datetime]:
    """
    When possible return the modified date

    return: a datetime or None
    """
    stat = path.stat()
    if stat is None:
        return None
    result = datetime.datetime.fromtimestamp(stat.st_mtime, tz=tzlocal())
    return result


def gswrap_newer(has_dest: List[Path], src_prefix: str, dest_prefix: str) -> List[Path]:
    """
    Get a list of objects where they are newer on the source

    has_dest: list of Path suffixes that are present in both the src and dest prefix
    src_prefix: string of the source gs:// or /fs
    dest_prefix: string of the dest gs:// or /fs
    """
    # TODO make this threaded if needed
    newer = []
    GS_PREFIX = "gs://"
    for cur_suffix in has_dest:
        if src_prefix.startswith(GS_PREFIX):
            gcs_file_path_str = f"{src_prefix}{cur_suffix}"
            src_date = get_modified_date_or_none_gcs(url=gcs_file_path_str)
        else:
            src_path = (Path(src_prefix) / cur_suffix).resolve()
            src_date = get_modified_date_or_none_fs(path=src_path)
        if dest_prefix.startswith(GS_PREFIX):
            gcs_file_path_str = f"{dest_prefix}{cur_suffix}"
            dest_date = get_modified_date_or_none_gcs(url=gcs_file_path_str)
        else:
            dest_path = (Path(dest_prefix) / cur_suffix).resolve()
            dest_date = get_modified_date_or_none_fs(path=dest_path)
        if (
            not isinstance(dest_date, datetime.datetime)
            or not isinstance(src_date, datetime.datetime)
            or src_date is None
            or dest_date is None
        ):
            logger.info(
                f"Unable to determine dates for {cur_suffix} at {src_prefix} and {dest_prefix}"
            )
            newer.append(cur_suffix)
            continue
        if dest_date == src_date:
            # skipping
            pass
        elif dest_date > src_date:
            # skipping
            pass
        else:
            newer.append(cur_suffix)
    return newer


def gs_wrap_sync(
    push: bool,
    bucket_name: str,
    file_system_root: Path,
    folder_name: str,
    only_copy_newer: bool = True,
    bucket_prefix_folder: str = "",
) -> None:
    """
    Uses gs_wrap see: https://gs-wrap.readthedocs.io/en/latest/readme.html

    get recursive list of files in both locations
    see:
    https://gs-wrap.readthedocs.io/en/latest/readme.html#list-objects-in-your-bucket

    first list, file exists in source but absent in dest
    second list, file exists in both

    use sets

    if only copy newer
    stat files in second list to see if source is newer, filter is only newer in source
    see:
    https://gs-wrap.readthedocs.io/en/latest/readme.html#copy-os-stat-of-a-file-or-metadata-of-a-blob

    create missing directories in dest

    copy all files in first and second list see:
    https://gs-wrap.readthedocs.io/en/latest/readme.html#perform-multiple-copy-operations-in-one-call

    push: Defines direction
    bucket_name: string without the gs:// prefix
    file_system_root: Path object of the root
    folder_name: where to sync between the two roots
    only_copy_newer: decide if to test if files are newer
    bucket_prefix_folder: if you want a different root than the root of the bucket

    """
    parent_gs_url = f"gs://{bucket_name}/"
    full_gs_url = f"gs://{bucket_name}/{folder_name}/"
    if bucket_prefix_folder != "":
        parent_gs_url = f"gs://{bucket_name}/{bucket_prefix_folder}/"
        full_gs_url = f"gs://{bucket_name}/{bucket_prefix_folder}/{folder_name}/"
    bucket_files = list(client.ls(url=full_gs_url, recursive=True))
    fs_folder = file_system_root / Path(folder_name)

    if not len(bucket_files) and push:
        logger.info("Destination directory does not exist! Copying")
        client.cp(
            fs_folder,
            parent_gs_url,
            recursive=True,
            multithreaded=True,
            preserve_posix=True,
        )
        logger.info("Destination copied, for initial folder!")
        return
    else:
        logger.info(
            f"Cloud directory {full_gs_url} does exist! "
            + f"{len(bucket_files)} files and folders "
        )

    fs_files = list(fs_folder.rglob("*"))
    fs_files_suffix = [i.relative_to(fs_folder) for i in fs_files]

    bucket_suffix = [Path(*(i.split(f"{full_gs_url}")[1:])) for i in bucket_files]
    bad_suffixes = set([Path("/"), Path("."), Path("/.")])
    bad_suffixes = set([i for i in bucket_suffix if "." not in i.name]).union(
        bad_suffixes
    )

    bucket_suffix = [s for s in bucket_suffix if s not in bad_suffixes]
    fs_files_suffix = [s for s in fs_files_suffix if s not in bad_suffixes]

    src, dest = (
        (fs_files_suffix, bucket_suffix) if push else (bucket_suffix, fs_files_suffix)
    )
    no_dest = []
    has_dest = []

    for cur_suffix in src:
        if cur_suffix in dest:
            has_dest.append(cur_suffix)
        else:
            no_dest.append(cur_suffix)

    copy_suffixes = []

    if not only_copy_newer:
        copy_suffixes = has_dest + no_dest
    else:
        if push:
            copy_suffixes = gswrap_newer(has_dest, f"{fs_folder}", f"{full_gs_url}")
        else:
            copy_suffixes = gswrap_newer(has_dest, f"{full_gs_url}", f"{fs_folder}")
        copy_suffixes += no_dest

    if not copy_suffixes:
        logger.info("No files to copy!")
        return

    if push:
        to_cloud = []
        for cur_suffix in copy_suffixes:
            source_path = fs_folder / cur_suffix
            dest_str = f"{full_gs_url}{cur_suffix}"
            to_cloud.append((source_path, dest_str))
        logger.info(
            f"Copying {len(to_cloud)} items from"
            + f" for example({source_path} to {dest_str})"
        )
        client.cp_many_to_many(
            srcs_dsts=tuple(to_cloud),
            multithreaded=False,
            recursive=True,
            preserve_posix=True,
        )
    else:
        from_cloud = []
        for cur_suffix in copy_suffixes:
            source_str = f"{full_gs_url}{cur_suffix}"
            dest_path = fs_folder / cur_suffix
            from_cloud.append((source_str, dest_path))
        logger.info(
            f"Copying {len(from_cloud)} items "
            + f"from for example({source_str} to {dest_path})"
        )
        client.cp_many_to_many(
            srcs_dsts=tuple(from_cloud),
            multithreaded=False,
            recursive=True,
        )
    return


def gsutil_sync(
    push: bool,
    bucket_name: str,
    file_system_root: Path,
    folder_name: str,
    max_server_error_attempts: int = 100,
    retry_interval_seconds: int = 5,
    only_copy_newer: bool = True,
    bucket_prefix_folder: str = "",
) -> None:
    """
    Either pushes or pulls newer files to or from Google Cloud Storage

    returns: None
    """
    fs_dir_path = file_system_root / folder_name
    if not fs_dir_path.is_dir():
        fs_dir_path.mkdir(parents=True)
    for attempts in range(1, max_server_error_attempts + 1):
        logger.info(f"Start of {attempts} of {max_server_error_attempts} GCS sync")
        try:
            timer001 = datetime.datetime.now()
            gs_wrap_sync(
                push=push,
                bucket_name=bucket_name,
                file_system_root=file_system_root,
                folder_name=folder_name,
                only_copy_newer=only_copy_newer,
                bucket_prefix_folder=bucket_prefix_folder,
            )
            timer002 = datetime.datetime.now()
            sync_duration = timer002 - timer001
            logger.info(f"Sync Complete! -> Sync Duration: {sync_duration}")
        except (
            ServerError,
            ReadTimeout,
            ConnectTimeout,
            ConnectionError,
            TooManyRedirects,
            InvalidURL,
            ChunkedEncodingError,
            ContentDecodingError,
            SSLError,
            ProxyError,
            HTTPError,
            MissingSchema,
            RetryError,
            MissingSchema,
            RetryError,
        ) as e:
            sleep_interval = 5.0
            logger.info(
                f"{attempts} of {max_server_error_attempts} "
                + "Server Error Exception Caught, sleeping for "
                + f"{sleep_interval} seconds"
            )
            exception_string = repr(e)
            exception_type_string = repr(type(e))
            exception_args_string = repr(e.args)
            logger.info(
                "Here is the exception info: "
                + f"\n type: {exception_type_string}"
                + f"\n args: {exception_args_string}"
                + f"\n exception: {exception_string}"
            )
            time.sleep(sleep_interval)
            continue
        break
    pretty_bad = max_server_error_attempts / 4
    if attempts > pretty_bad:
        logger.info(
            "Link to CGS is pretty bad " + f"({attempts} > {pretty_bad} attempts)!"
        )
    if attempts > max_server_error_attempts:
        logger.error(
            "Link to CGS is not really usable "
            + f"({attempts} beyond {max_server_error_attempts} threshold)!!!"
        )
        raise Exception("Link to CGS is not stable!!!")
    logger.info(
        f"All Done! GCS {attempts} attempts from a possible "
        + f"{max_server_error_attempts} GCS syncs"
    )
