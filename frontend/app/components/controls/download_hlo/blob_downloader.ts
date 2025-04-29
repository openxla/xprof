import {Injectable} from '@angular/core';
import {objectUrlFromSafeSource, unwrapUrl} from 'safevalues';
import {setHref} from 'safevalues/dom';

/**
 * Service for downloading a client-side generated file to the user's machine.
 *
 * Example usage:
 * blobDownloader.downloadString(
 *     '1,2', 'myFile.csv', 'text/csv;charset=utf-8;');
 */
@Injectable({providedIn: 'root'})
export class BlobDownloader {
  /**
   * Download the specified text to the user's machine as a file.
   * Set type to application/octet-stream to enable the safe download.
   */
  downloadString(
      text: string, fileName: string, type = 'application/octet-stream'): void {
    this.downloadBlob(new Blob([text], {type}), fileName);
  }

  /**
   * Download the specified Blob content to the user's machine as a file.
   * To enable the safe download, the blob needs to be in type of
   * application/octet-stream.
   */
  downloadBlob(blob: Blob, fileName: string): void {
    const downloadLink = document.createElement('a');
    const objectUrl = objectUrlFromSafeSource(blob);
    setHref(downloadLink, objectUrl);

    if (this.isDownloadSupported(downloadLink)) {
      downloadLink.download = fileName;
    }

    downloadLink.style.display = 'none';

    document.body.appendChild(downloadLink);
    downloadLink.click();
    document.body.removeChild(downloadLink);

    // In mobile WebKit, where the download attribute is not yet supported,
    // calling URL.revokeObjectURL prevents the user from opening the page in
    // browser.  Where download is supported, we can safely call
    // URL.revokeObjectURL after clicking.
    if (this.isDownloadSupported(downloadLink)) {
      URL.revokeObjectURL(unwrapUrl(objectUrl));
    }
  }

  /**
   * Gets whether the anchor element supports the download attribute.
   *
   * https://developer.mozilla.org/en-US/docs/Web/HTML/Element/a#Attributes
   */
  private isDownloadSupported(element: HTMLAnchorElement) {
    return element.download !== undefined;
  }
}
